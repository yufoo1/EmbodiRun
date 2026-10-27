"""CLI for Go2 RealSense -> onboard NX ActiveVLN -> full-chunk motion."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from embodirun.model_services.backends.embodiinfer.compat import connect_client
from embodirun.robots.unitree.go2.camera import Go2CameraClient
from embodirun.robots.unitree.go2.client import Go2ControlClient
from embodirun.robots.unitree.go2.navigation.async_chunk_session import AsyncChunkNavigationSession, OverlapConfig
from embodirun.robots.unitree.go2.navigation.chunk_session import ChunkNavigationSession, ChunkSessionConfig

from .policy import ActiveVLNPolicy, check_capabilities


@dataclass(frozen=True)
class NavigationConfig:
    """Connection placeholders and bounded motion settings, with secrets in env vars."""

    camera_url: str = ""
    control_url: str = ""
    inference_url: str = ""
    robot_id: str = "go2"
    camera_token_env: str = "GO2_CAMERA_TOKEN"
    control_token_env: str = "GO2_API_TOKEN"
    inference_token_env: str = "ACTIVEVLN_BENCHMARK_TOKEN"
    inference_timeout_s: float = 60.0
    camera_timeout_s: float = 2.0
    control_timeout_s: float = 0.5
    mode: str = "overlap"
    session: ChunkSessionConfig = field(default_factory=ChunkSessionConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)

    @classmethod
    def load(cls, path: Path) -> NavigationConfig:
        """Read strict JSON without connecting, installing, or changing the system."""
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("configuration must be a JSON object")
        session = ChunkSessionConfig(**value.pop("session", {}))
        overlap = OverlapConfig(**value.pop("overlap", {}))
        config = cls(**value, session=session, overlap=overlap)
        if config.mode not in ("chunk", "overlap"):
            raise ValueError("mode must be chunk or overlap")
        for name in ("inference_timeout_s", "camera_timeout_s", "control_timeout_s"):
            number = getattr(config, name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("camera_url", "control_url", "inference_url"):
            value = getattr(config, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a URL or an empty placeholder")
            parsed = urlsplit(value)
            if value and (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in ("", "/")
            ):
                raise ValueError(f"{name} must be an HTTP service root without credentials")
        return config

    def require_endpoints(self) -> None:
        """Fail before any network access while a deployment address is blank."""
        missing = [name for name in ("camera_url", "control_url", "inference_url") if not getattr(self, name)]
        if missing:
            raise ValueError("fill deployment endpoints first: " + ", ".join(missing))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodirun go2 activevln", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("plan", "inspect configuration offline; blank endpoints are allowed"),
        ("check", "read camera, robot state and model capabilities without motion"),
        ("run", "run navigation; defaults to one observation/inference without motion"),
    ):
        command = sub.add_parser(name, help=description)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--instruction", required=True)
            command.add_argument("--execute", action="store_true", help="enable motion on an already armed agent")
            command.add_argument("--output-jsonl", type=Path, help="create a new event log; never overwrite")
    sub.add_parser("edge", help="forward to the existing Go2 probe/install/start/status/stop CLI")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Connect existing edge/model services and run navigation with explicit motion opt-in."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "edge":
        from ..streamvln.cli import main as edge_main

        return edge_main(arguments[1:])
    args = _parser().parse_args(arguments)
    try:
        config = NavigationConfig.load(args.config)
        if args.command == "plan":
            print(json.dumps({"configuration": asdict(config), "motion_enabled": False}, indent=2))
            return 0
        config.require_endpoints()
        if args.command == "run" and not args.instruction.strip():
            raise ValueError("instruction must be non-empty")
        client = connect_client(
            config.inference_url,
            os.environ.get(config.inference_token_env),
            timeout_s=config.inference_timeout_s,
        )
        capabilities = check_capabilities(client, overlap=config.mode == "overlap")
        camera = Go2CameraClient(
            config.camera_url,
            token=os.environ.get(config.camera_token_env),
            timeout_s=config.camera_timeout_s,
        )
        base = Go2ControlClient(
            config.control_url,
            token=os.environ.get(config.control_token_env),
            timeout_s=config.control_timeout_s,
        )
        if args.command == "check":
            state = base.state()
            observation = camera.capture(episode_id="go2-check", reset=True)
            print(
                json.dumps(
                    {
                        "camera_sequence": observation.sequence,
                        "depth_available": observation.depth is not None,
                        "state_sequence": state.sequence,
                        "active_lease": state.active_velocity_lease_id,
                        "inference": capabilities,
                        "motion_enabled": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        with ExitStack() as stack:
            output = None
            output_lock = threading.Lock()
            if args.output_jsonl:
                args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
                output = stack.enter_context(args.output_jsonl.open("x", encoding="utf-8"))

            def emit(event: dict[str, object]) -> None:
                line = json.dumps(event, ensure_ascii=False, allow_nan=False)
                with output_lock:
                    print(line, flush=True)
                    if output is not None:
                        output.write(line + "\n")
                        output.flush()

            policy = ActiveVLNPolicy(client, robot_id=config.robot_id, event_sink=emit)
            stack.callback(policy.close)

            def interrupt(_signum: int, _frame: object) -> None:
                raise KeyboardInterrupt

            old_handler = signal.signal(signal.SIGTERM, interrupt)
            stack.callback(signal.signal, signal.SIGTERM, old_handler)
            if config.mode == "overlap":
                session = AsyncChunkNavigationSession(
                    policy, camera, base, config=config.session, overlap=config.overlap, event_sink=emit
                )
            else:
                session = ChunkNavigationSession(policy, camera, base, config=config.session, event_sink=emit)
            session.run(args.instruction, execute=args.execute)
        return 0
    except KeyboardInterrupt:
        print("navigation interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"navigation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
