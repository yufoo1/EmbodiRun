"""ActiveVLN action contract, CLI, and real localhost HTTP wiring."""

import base64
import hashlib
import json
import threading
import time
from dataclasses import replace
from email import policy as email_policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from embodirun.bindings.unitree.go2.activevln.cli import NavigationConfig, main
from embodirun.bindings.unitree.go2.activevln.policy import (
    ACTION_SPACE,
    ActiveVLNPolicy,
    check_capabilities,
    navigation_chunk,
)
from embodirun.model_services.contracts import PolicyAction, PolicyResult, Session
from embodirun.robots.unitree.go2.navigation.frames import EncodedRGBFrame
from embodirun.robots.unitree.go2.navigation.observation import NavigationObservation
from embodirun.robots.unitree.go2.navigation.request import NavigationRequest


def result(rows, **values):
    return PolicyResult(
        "request",
        "session",
        0,
        1,
        ACTION_SPACE,
        (PolicyAction("discrete_chunk", {"parsed_valid": True, "rows": rows, **values}),),
    )


def test_action_units_stop_and_padding():
    chunk = navigation_chunk(result([[1, 75], [3, 45], [0, 0]]), 7)
    assert chunk.observation_sequence == 7 and chunk.terminal
    assert chunk.commands[0].distance_m == 0.75
    assert chunk.commands[1].angle_deg == 45
    assert chunk.commands[1].kind.value == "turn_right"
    assert len(navigation_chunk(result([[0, 0], [1, 25], [-1, 0]]), 7).commands) == 1
    assert len(navigation_chunk(result([[1, 50], [-1, 0], [-1, 0]]), 7).commands) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [[1, 25]],
        [[-1, 0]] * 3,
        [[1, True], [-1, 0], [-1, 0]],
        [[1, float("nan")], [-1, 0], [-1, 0]],
        [[1.5, 25], [-1, 0], [-1, 0]],
        [[1, 100], [-1, 0], [-1, 0]],
        [[2, 90], [-1, 0], [-1, 0]],
        [[0, 25], [-1, 0], [-1, 0]],
        [[-1, 25], [-1, 0], [-1, 0]],
        [[1, 25], [-1, 0], [2, 15]],
        ["12", [-1, 0], [-1, 0]],
    ],
)
def test_malformed_chunks_are_rejected(rows):
    with pytest.raises((TypeError, ValueError)):
        navigation_chunk(result(rows), 1)


def test_invalid_parser_result_and_wrong_action_space_rejected():
    valid = result([[1, 25], [-1, 0], [-1, 0]])
    for wrong in (
        result([[1, 25], [-1, 0], [-1, 0]], parsed_valid=False),
        replace(valid, action_space="streamvln.action_chunk.v1"),
        replace(valid, actions=(PolicyAction("action_chunk", valid.actions[0].values),)),
    ):
        with pytest.raises(ValueError):
            navigation_chunk(wrong, 1)


class Client:
    def __init__(self):
        self.steps = []
        self.closed = []
        self.revision_offset = 0
        self.adapter = {"model": "activevln", "recurrent": True, "max_batch_size": 1, "action_space": ACTION_SPACE}

    def capabilities(self):
        return {"adapter": self.adapter}

    def open_session(self, **kwargs):
        self.opened = kwargs
        return Session("session", 0)

    def step(self, observation):
        self.steps.append(observation)
        return replace(
            result([[1, 25], [1, 25], [3, 15]]),
            request_id=observation.request_id,
            step_id=observation.step_id,
            session_revision=len(self.steps) + self.revision_offset,
        )

    def close(self, session_id):
        self.closed.append(session_id)


def request(sequence=1, reset=True):
    return NavigationRequest(
        "go to door",
        NavigationObservation(
            "episode",
            sequence,
            reset,
            (EncodedRGBFrame(sequence, float(sequence), b"\xff\xd8jpeg"),),
        ),
    )


def test_policy_orders_recurrent_steps_preserves_image_and_closes():
    client = Client()
    policy = ActiveVLNPolicy(client)
    assert len(policy.plan(request()).commands) == 3
    policy.plan(request(2, False))
    policy.close()
    assert client.closed == ["session"]
    assert [r.step_id for r in client.steps] == [0, 1]
    assert client.steps[0].images[0].data == b"\xff\xd8jpeg"
    assert client.steps[0].state == {"benchmark_slot": 0}
    assert not any("async" in r.metadata for r in client.steps)
    assert client.opened["metadata"] == {"episode_id": "episode"}


def test_revision_failure_poisons_session_and_still_releases_it():
    client = Client()
    client.revision_offset = 2
    policy = ActiveVLNPolicy(client)
    with pytest.raises(ValueError, match="revision"):
        policy.plan(request())
    with pytest.raises(RuntimeError, match="cannot be reused"):
        policy.plan(request(2, False))
    policy.close()
    assert client.closed == ["session"] and len(client.steps) == 1


def test_overlap_requires_capability_and_matching_acknowledgement():
    client = Client()
    with pytest.raises(ValueError, match="execution-prefix-aware"):
        check_capabilities(client, overlap=True)
    client.adapter["execution_prefix"] = "activevln.execution_prefix.v1"
    check_capabilities(client, overlap=True)
    policy = ActiveVLNPolicy(client)
    chunk = policy.plan(request())
    with pytest.raises(ValueError, match="acknowledge"):
        policy.plan_after(request(2, False), chunk.commands[:1])
    assert client.steps[-1].metadata["activevln_execution"] == {
        "schema": "activevln.execution_prefix.v1",
        "previous_step_id": 0,
        "remaining_actions": [[1, 25.0]],
    }
    with pytest.raises(RuntimeError, match="cannot be reused"):
        policy.plan(request(3, False))
    policy.close()
    assert client.closed == ["session"]


@pytest.mark.parametrize("changes", [{"max_batch_size": 4}, {"model": "pi05"}, {"action_space": "other"}])
def test_capability_mismatch_fails_before_session(changes):
    client = Client()
    client.adapter.update(changes)
    with pytest.raises(ValueError):
        check_capabilities(client)
    assert not hasattr(client, "opened")


def test_blank_ip_configuration_is_offline_and_run_fails_before_network(monkeypatch, capsys):
    config = Path(__file__).parents[1] / "configs/activevln/go2-orin-nx.json"
    import embodirun.bindings.unitree.go2.activevln.cli as cli

    monkeypatch.setattr(cli, "connect_client", lambda *a, **k: pytest.fail("unexpected network access"))
    assert main(["plan", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["motion_enabled"] is False
    assert main(["run", "--config", str(config), "--instruction", "go"]) == 1
    assert "fill deployment endpoints" in capsys.readouterr().err


def test_motion_cannot_be_enabled_by_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"execute": True}))
    with pytest.raises(TypeError):
        NavigationConfig.load(path)


def test_edge_install_resolves_the_current_package_layout():
    from embodirun.bindings.unitree.go2.streamvln.deployer import Go2AgentDeployer

    class Transport:
        def __init__(self):
            self.uploaded = []

        def resolve_path(self, path):
            return "/tmp/go2-agent"

        def make_dirs(self, *args, **kwargs):
            pass

        def put_file(self, local, remote):
            self.uploaded.append(remote)

        def put_bytes(self, *args, **kwargs):
            pass

    transport = Transport()
    manifest = Go2AgentDeployer(transport, "go2-agent").install()
    assert manifest["file_count"] == len(transport.uploaded)
    assert "/tmp/go2-agent/src/embodirun/robots/unitree/go2/agent/camera/realsense.py" in transport.uploaded
    assert "/tmp/go2-agent/src/embodirun/bindings/unitree/go2/activevln/cli.py" in transport.uploaded


def test_edge_alias_reuses_existing_deployment_cli(monkeypatch):
    from embodirun.bindings.unitree.go2.streamvln import cli

    called = []
    monkeypatch.setattr(cli, "main", lambda argv: called.append(argv) or 0)
    assert main(["edge", "--host", "robot", "--username", "user", "status"]) == 0
    assert called == [["--host", "robot", "--username", "user", "status"]]


@pytest.mark.parametrize(
    "namespace,execute,mode",
    [
        ("embodiinfer", False, "chunk"),
        ("vvla", True, "chunk"),
        ("embodiinfer", True, "overlap"),
        ("vvla", True, "overlap"),
    ],
)
def test_cli_end_to_end_over_local_http(namespace, execute, mode, tmp_path, monkeypatch, capsys):
    """Real HTTP transport, virtual robot/camera/model; never loads a checkpoint."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for name in ("GO2_API_TOKEN", "GO2_CAMERA_TOKEN", "ACTIVEVLN_BENCHMARK_TOKEN"):
        monkeypatch.setenv(name, "t" * 32)
    calls, writes, inference, frames = [], [], [], []
    active_until = 0.0
    camera_sequence = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            nonlocal camera_sequence
            assert self.headers["Authorization"] == "Bearer " + "t" * 32
            calls.append(("GET", self.path))
            if self.path == "/healthz":
                self.reply({"ok": True})
            elif self.path == "/v1/capabilities":
                self.reply(
                    {
                        "schema": f"{namespace}.policy.capabilities.v1",
                        "adapter": {**Client().adapter, "execution_prefix": "activevln.execution_prefix.v1"},
                    }
                )
            elif self.path == "/v1/state":
                self.reply(
                    {
                        "transport": "unitree-sdk2",
                        "robot_state_available": True,
                        "robot_state_fresh": True,
                        "operator_motion_ready": True,
                        "capabilities": {"stream_lease_renewal": True},
                        "active_action": {"id": "lease"} if time.monotonic() < active_until else None,
                        "robot_state": {"position": [0, 0], "velocity": [0, 0], "yaw": 0, "yaw_rate": 0, "sequence": 1},
                    }
                )
            elif self.path == "/observation.json":
                if mode == "chunk":
                    assert time.monotonic() >= active_until
                camera_sequence += 1
                frames.append(camera_sequence)
                self.reply(
                    {
                        "navigation_ready": True,
                        "sequence": camera_sequence,
                        "captured_at_unix": time.time(),
                        "frame_age_s": 0.0,
                        "rgb": {
                            "sequence": camera_sequence,
                            "width": 640,
                            "height": 360,
                            "data_url": "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8test").decode(),
                        },
                        "depth": {
                            "sequence": camera_sequence,
                            "scale_m": 0.001,
                            "data_url": "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode(),
                        },
                    }
                )
            else:
                raise AssertionError(self.path)

        def do_POST(self):
            nonlocal active_until
            assert self.headers["Authorization"] == "Bearer " + "t" * 32
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(("POST", self.path))
            if self.path.endswith("/steps"):
                if mode == "chunk":
                    assert time.monotonic() >= active_until
                mime = BytesParser(policy=email_policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw
                )
                parts = list(mime.iter_parts())
                metadata = json.loads(parts[0].get_payload(decode=True))
                assert metadata["schema"] == f"{namespace}.policy.step.v1"
                assert metadata["state"] == {"benchmark_slot": 0}
                assert parts[1].get_payload(decode=True) == b"\xff\xd8test"
                step = metadata["step_id"]
                inference.append(metadata)
                rows = (
                    [[1, 25], [-1, 0], [-1, 0]]
                    if step < (2 if mode == "overlap" else 1)
                    else [[0, 0], [-1, 0], [-1, 0]]
                )
                values = {"parsed_valid": True, "rows": rows}
                if mode == "overlap" and step > 0:
                    prefix = metadata["metadata"]["activevln_execution"]
                    assert prefix["previous_step_id"] == step - 1
                    assert prefix["remaining_actions"]
                    assert time.monotonic() < active_until
                    time.sleep(0.15)
                    values["execution_prefix_sha256"] = hashlib.sha256(
                        json.dumps(prefix, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                self.reply(
                    {
                        "request_id": metadata["request_id"],
                        "session_id": "session",
                        "step_id": step,
                        "session_revision": step + 1,
                        "action_space": ACTION_SPACE,
                        "actions": [{"type": "discrete_chunk", "values": values}],
                    }
                )
            else:
                payload = json.loads(raw)
                if self.path == "/v1/sessions":
                    assert payload["schema"] == f"{namespace}.policy.session.v1"
                    self.reply({"session_id": "session", "session_revision": 0})
                elif self.path == "/v1/actions":
                    writes.append(payload)
                    if payload["action"] == "stream_move":
                        assert time.monotonic() >= active_until
                        active_until = time.monotonic() + payload["duration_s"]
                        self.reply({"action_id": "lease"})
                    else:
                        assert payload["action_id"] == "lease" and time.monotonic() < active_until
                        if "remaining_s" in payload:
                            active_until = time.monotonic() + payload["remaining_s"]
                        self.reply({"accepted": True})
                elif self.path == "/v1/stop":
                    writes.append({"action": "stop"})
                    active_until = 0
                    self.reply({"stopped": True})
                else:
                    raise AssertionError(self.path)

        def do_DELETE(self):
            calls.append(("DELETE", self.path))
            self.reply({"closed": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"camera_url": url, "control_url": url, "inference_url": url, "mode": mode}))
        from embodirun.services.host.cli.cli import main as unified_main

        def main(args):
            return unified_main(["go2", "activevln", *args])

        assert main(["check", "--config", str(config)]) == 0
        assert not writes and not inference and not any(method == "POST" for method, _ in calls)
        capsys.readouterr()
        output = tmp_path / "trace.jsonl"
        args = ["run", "--config", str(config), "--instruction", "go", "--output-jsonl", str(output)]
        assert main(args + (["--execute"] if execute else [])) == 0
        summary = json.loads(output.read_text().splitlines()[-1])
        assert summary["reason"] == ("stop" if execute else "dry_run")
        assert ("DELETE", "/v1/sessions/session") in calls
        if execute:
            assert len(inference) == (3 if mode == "overlap" else 2)
            assert [w["action"] for w in writes].count("stream_move") == 1
            assert writes[-1] == {"action": "stop"}
            updates = [w for w in writes if w["action"] == "update_move"]
            if mode == "chunk":
                assert updates[-1]["vx"] == 0
                assert all(w["vx"] > 0 for w in updates[:-1])
            else:
                assert all(w["vx"] > 0 for w in updates)
                assert summary["mode"] == "overlap"
        else:
            assert not writes and len(inference) == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
