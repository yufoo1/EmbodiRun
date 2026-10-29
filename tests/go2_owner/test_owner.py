"""Unified owner lifecycle, HTTP isolation and policy-independent deployment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from embodirun.bindings.unitree.go2.activevln.cli import NavigationConfig
from embodirun.deployment.go2.deployer import Go2OwnerDeployer
from embodirun.deployment.go2.transport import CommandResult
from embodirun.robots.mobile_base.motion import PlanarVelocityCommand
from embodirun.robots.unitree.go2.camera import Go2CameraClient, Go2CameraError
from embodirun.robots.unitree.go2.client import Go2ClientError, Go2ControlClient
from embodirun_go2_owner.camera.frame_store import FrameStore
from embodirun_go2_owner.camera.observation import CameraObservationService
from embodirun_go2_owner.camera.types import VideoProfile
from embodirun_go2_owner.config import OwnerConfig
from embodirun_go2_owner.control.transport import DryRunTransport
from embodirun_go2_owner.runtime import Go2Owner
from embodirun_go2_owner.server import OwnerHTTPServer, serve

REPO = Path(__file__).resolve().parents[2]

MOVE = {"action": "stream_move", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration_s": 5}


class FakeCamera:
    max_frame_age = 1
    depth_store = None

    def __init__(self):
        self.frame_store = FrameStore()
        self.frame_store.set_source_profile("fake", VideoProfile(640, 480, "MJPG", 0, 0, 15))
        self.observations = CameraObservationService(self)
        self.closed = False

    def start(self):
        self.frame_store.set_capture_running(True)
        self.frame_store.update(b"\xff\xd8fake-jpeg\xff\xd9")

    def close(self):
        self.closed = True
        self.frame_store.set_capture_running(False)


def config(tmp_path, monkeypatch, *, ready=False, value=None):
    monkeypatch.delenv("GO2_API_TOKEN", raising=False)
    path = tmp_path / "owner.json"
    path.write_text(json.dumps(value or {"control": {"mode": "dry-run"}}))
    return replace(OwnerConfig.load(path, operator_ready=ready), port=0)


@pytest.fixture
def owner_server(tmp_path, monkeypatch):
    settings = config(tmp_path, monkeypatch, ready=True)
    owner = Go2Owner(settings, camera=FakeCamera())
    server = OwnerHTTPServer(owner)
    owner.start()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield owner, f"http://127.0.0.1:{server.server_port}"
    finally:
        owner.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(url, path, payload=None, token=None):
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    try:
        response = urlopen(Request(url + path, data=data, headers=headers), timeout=2)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def test_one_endpoint_serves_state_rgb_and_stop_with_shared_auth(owner_server):
    owner, url = owner_server
    owner.config = replace(owner.config, token="a" * 32)
    for path, payload in (("/v1/state", None), ("/observation/rgb.json", None), ("/v1/stop", {})):
        assert request(url, path, payload)[0] == 401
        assert request(url, path, payload, "a" * 32)[0] == 200
    observation = Go2CameraClient(url, token="a" * 32, require_depth=False).capture(episode_id="rgb", reset=True)
    assert observation.depth is None
    assert observation.rgb_frames[0].width == 640
    assert observation.metadata["capture_uncertainty_s"] >= 0
    with pytest.raises(Go2CameraError, match="aligned depth capture is disabled"):
        Go2CameraClient(url, token="a" * 32).capture(episode_id="rgbd", reset=True)


def test_blocked_camera_response_cannot_block_stop_or_state(owner_server, monkeypatch):
    owner, url = owner_server
    entered, release = threading.Event(), threading.Event()
    original = owner.camera.observations.rgb_observation_payload

    def blocked():
        entered.set()
        assert release.wait(3)
        owner.camera.frame_store.update(b"\xff\xd8fake\xff\xd9")
        return original()

    monkeypatch.setattr(owner.camera.observations, "rgb_observation_payload", blocked)
    assert request(url, "/v1/actions", MOVE)[0] == 202
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(request, url, "/observation/rgb.json")
        try:
            assert entered.wait(1)
            assert request(url, "/v1/state")[0] == 200
            started = time.monotonic()
            status, payload = request(url, "/v1/stop", {})
            assert time.monotonic() - started < 1
            assert status == 200 and payload["operator_motion_ready"] is False
            assert owner.executor.transport.state().velocity == (0.0, 0.0, 0.0)
            assert request(url, "/v1/actions", MOVE)[0] == 503
        finally:
            release.set()
        assert pending.result()[0] == 200


def test_stale_or_stopped_capture_is_rejected_without_disabling_control(owner_server):
    owner, url = owner_server
    owner.camera.frame_store.set_capture_running(False)
    assert request(url, "/observation/rgb.json")[0] == 503
    assert request(url, "/v1/state")[0] == 200
    owner.camera.frame_store.set_capture_running(True)
    owner.camera.frame_store.update(b"\xff\xd8fake\xff\xd9", captured_at_monotonic=time.monotonic() - 5)
    assert request(url, "/observation/rgb.json")[0] == 503


def test_shutdown_stops_motors_before_camera_cleanup_and_is_idempotent(tmp_path, monkeypatch):
    settings = config(tmp_path, monkeypatch, ready=True)
    camera = FakeCamera()
    transport = DryRunTransport()
    owner = Go2Owner(settings, transport=transport, camera=camera)
    owner.start()
    owner.executor.execute(MOVE)
    closed = []

    def close_camera():
        assert transport.state().velocity == (0.0, 0.0, 0.0)
        assert not owner.executor.operator_motion_ready
        closed.append(True)

    monkeypatch.setattr(camera, "close", close_camera)
    owner.close()
    owner.close()
    assert closed == [True]


@pytest.mark.parametrize("failure", ["bind", "camera"])
def test_start_failure_cleans_up_control_and_camera(tmp_path, monkeypatch, failure):
    import embodirun_go2_owner.server as server_module

    camera = FakeCamera()
    owner = Go2Owner(config(tmp_path, monkeypatch, ready=True), camera=camera)

    def fail(*args):
        raise OSError("startup failure")

    if failure == "bind":
        monkeypatch.setattr(server_module, "OwnerHTTPServer", fail)
    else:
        monkeypatch.setattr(camera, "start", fail)
    with pytest.raises(OSError, match="startup failure"):
        serve(owner, install_signal_handlers=False)
    assert camera.closed
    assert not owner.executor.operator_motion_ready


def test_readiness_is_explicit_and_not_persisted_in_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("GO2_OPERATOR_READY", "1")
    owner = Go2Owner(config(tmp_path, monkeypatch))
    try:
        assert not owner.executor.snapshot()["operator_motion_ready"]
    finally:
        owner.close()
    with pytest.raises(ValueError, match="readiness"):
        config(tmp_path, monkeypatch, value={"control": {"operator_ready": True}})


def test_camera_config_uses_one_endpoint_and_ignores_legacy_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GO2_CAMERA_PORT", "9999")
    monkeypatch.setenv("GO2_CAMERA_TOKEN", "unused-legacy-token")
    settings = config(tmp_path, monkeypatch, value={"server": {"port": 8081}, "camera": {"backend": "v4l2"}})
    assert not hasattr(settings.control, "port")
    assert not hasattr(settings.camera, "port")
    assert not hasattr(settings.camera, "token")


def test_single_robot_url_and_conflicting_legacy_endpoints(tmp_path):
    path = tmp_path / "navigation.json"
    path.write_text(json.dumps({"robot_url": "http://127.0.0.1:8080"}))
    settings = NavigationConfig.load(path)
    assert settings.camera_url == settings.control_url == settings.robot_url
    path.write_text(json.dumps({"robot_url": "http://localhost:8080", "camera_url": "http://localhost:8765"}))
    with pytest.raises(ValueError, match="conflicts"):
        NavigationConfig.load(path)


class FakeTransport:
    def __init__(self):
        self.files, self.commands = {}, []

    def resolve_path(self, path):
        return "/home/unitree/" + path

    def make_dirs(self, path, **kwargs):
        pass

    def put_file(self, local_path, remote_path, **kwargs):
        self.files[remote_path] = local_path.read_bytes()

    def put_bytes(self, payload, remote_path, **kwargs):
        self.files[remote_path] = payload

    def run(self, command, **kwargs):
        self.commands.append(command)
        return CommandResult(0, "running\t123\n")


def test_deployment_uploads_only_owner_and_uses_one_supervised_process(tmp_path):
    transport = FakeTransport()
    deployer = Go2OwnerDeployer(transport, "owner")
    manifest = deployer.install()
    assert manifest["package"] == "embodirun_go2_owner"
    assert not any("/src/embodirun/" in name for name in transport.files)
    config_path = tmp_path / "owner.json"
    config_path.write_text('{"control": {"mode": "dry-run"}}')
    result = deployer.start(
        config_path,
        python="/usr/bin/python3.8",
        supervisor_python="/opt/host/bin/python",
        token="a" * 32,
        sdk_pythonpath="/vendor/sdk",
    )
    assert list(result["services"]) == ["go2"]
    validation, start = transport.commands[-2:]
    assert validation.argv[-1] == "--validate"
    assert start.argv[0] == "/opt/host/bin/python"
    payload = json.loads(start.stdin)
    assert payload["argv"][:3] == ["/usr/bin/python3.8", "-m", "embodirun_go2_owner"]
    assert "--operator-ready" not in payload["argv"]
    assert payload["environment"]["GO2_API_TOKEN"] == "a" * 32
    assert payload["environment"]["PYTHONPATH"].endswith(":/vendor/sdk")
    assert not any(b"a" * 32 in data for data in transport.files.values())
    deployer.status("/opt/host/bin/python")
    deployer.stop("/opt/host/bin/python")
    assert all(
        command.argv[command.argv.index("--service-id") + 1] == "go2"
        for command in transport.commands
        if "--service-id" in command.argv
    )


def test_owner_imports_and_validation_need_no_core_or_native_dependencies():
    code = """
import sys
from embodirun_go2_owner.cli import main
assert main(['--config', sys.argv[1], '--validate']) == 0
assert not {'embodirun', 'torch', 'pyrealsense2', 'unitree_sdk2py'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(REPO / "configs/go2/owner-dry-run.json")],
        env=dict(os.environ, PYTHONPATH=str(REPO / "integrations/go2_owner/src")),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_motion_watchdog_still_stops_without_heartbeats(owner_server, monkeypatch):
    from embodirun_go2_owner.control import lease

    monkeypatch.setattr(lease, "STREAM_HEARTBEAT_TIMEOUT_S", 0.1)
    owner, url = owner_server
    assert request(url, "/v1/actions", MOVE)[0] == 202
    deadline = time.monotonic() + 1
    while owner.executor.snapshot()["active_action"] is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert owner.executor.snapshot()["active_action"] is None
    assert owner.executor.transport.state().velocity == (0.0, 0.0, 0.0)
    assert owner.executor.snapshot()["operator_motion_ready"] is False


def test_http_client_releases_own_lease_and_can_start_next_task(owner_server):
    owner, url = owner_server
    client = Go2ControlClient(url, expected_transport="dry-run", timeout_s=2)
    for _ in range(2):
        client.preflight(require_release=True)
        action_id = client.start_velocity_lease(PlanarVelocityCommand(0.1, 0, 0), duration_s=5)
        client.release_velocity_lease(action_id)
        assert client.state().active_velocity_lease_id is None
        assert owner.executor.operator_motion_ready
    client.stop()
    with pytest.raises(Go2ClientError):
        client.release_velocity_lease(action_id)
    assert not owner.executor.operator_motion_ready


def test_current_python38_owner_starts_and_handles_sigterm(tmp_path):
    import shutil
    import socket

    # Exercise the actual vendor-compatible interpreter when available on macOS.
    python = "/usr/bin/python3"
    if not Path(python).exists():
        python = shutil.which("python3.8")
    if python is None:
        pytest.skip("Python 3.8 interpreter is not available")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    path = tmp_path / "owner.json"
    path.write_text(json.dumps({"server": {"port": port}, "control": {"mode": "dry-run"}}))
    environment = dict(os.environ, PYTHONPATH=str(REPO / "integrations/go2_owner/src"))
    environment.pop("GO2_API_TOKEN", None)
    process = subprocess.Popen(
        [python, "-S", "-m", "embodirun_go2_owner", "--config", str(path)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                status, state = request(f"http://127.0.0.1:{port}", "/v1/state")
                break
            except OSError:
                if process.poll() is not None:
                    pytest.fail(process.communicate()[1])
                time.sleep(0.02)
        else:
            pytest.fail("owner did not start")
        assert status == 200 and state["operator_motion_ready"] is False
        process.terminate()
        _, error = process.communicate(timeout=5)
        assert process.returncode == 0, error
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_layer_boundaries_and_all_moved_navigation_modules_import():
    import importlib
    import pkgutil

    for name in ("embodirun.robots.mobile_base", "embodirun.application.navigation"):
        package = importlib.import_module(name)
        for module in pkgutil.walk_packages(package.__path__, name + "."):
            importlib.import_module(module.name)
    for root, forbidden in (
        (REPO / "src/embodirun/robots/unitree/go2", "embodirun.application"),
        (REPO / "src/embodirun/application/navigation", "robots.unitree.go2"),
        (REPO / "src/embodirun/deployment/go2", "bindings.unitree.go2"),
    ):
        for path in root.rglob("*.py"):
            assert forbidden not in path.read_text(), path


@pytest.mark.parametrize("backend", ["v4l2", "realsense"])
def test_capture_joins_real_worker_wrappers_without_opening_hardware(backend, monkeypatch):
    from embodirun_go2_owner.camera.capture import build_capture
    from embodirun_go2_owner.camera.settings import CameraServiceConfig

    capture = build_capture(
        CameraServiceConfig.from_mapping(
            {
                "backend": backend,
                "depth_scale": 0.001,
                "depth_calibrated": True,
            }
        )
    )
    for worker in capture.workers:
        monkeypatch.setattr(worker, "_run", worker._stop.wait)
    try:
        capture.start()
        assert all(worker.is_alive() for worker in capture.workers)
    finally:
        capture.close()
    assert not any(worker.is_alive() for worker in capture.workers)


def test_partial_capture_start_failure_joins_already_started_worker(tmp_path, monkeypatch):
    from embodirun_go2_owner.camera.capture import build_capture
    from embodirun_go2_owner.camera.settings import CameraServiceConfig

    capture = build_capture(
        CameraServiceConfig.from_mapping(
            {
                "backend": "v4l2",
                "depth_scale": 0.001,
                "depth_calibrated": True,
            }
        )
    )
    monkeypatch.setattr(capture.workers[0], "_run", capture.workers[0]._stop.wait)

    def fail():
        raise RuntimeError("depth worker startup failed")

    monkeypatch.setattr(capture.workers[1], "start", fail)
    owner = Go2Owner(config(tmp_path, monkeypatch, ready=True), camera=capture)
    with pytest.raises(RuntimeError, match="startup failed"):
        owner.start()
    assert not any(worker.is_alive() for worker in capture.workers)
    assert not owner.executor.operator_motion_ready
