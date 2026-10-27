"""Real worker/control concurrency without hardware, model weights or remote IO."""

import threading
import time
from dataclasses import replace

import pytest

from embodirun.robots.unitree.go2.navigation.async_chunk_session import (
    AsyncChunkNavigationSession,
    OverlapConfig,
    remaining_commands,
)
from embodirun.robots.unitree.go2.navigation.chunk_session import ChunkSessionConfig, NavigationChunk
from embodirun.robots.unitree.go2.navigation.discrete import NavigationCommand as Command
from embodirun.robots.unitree.go2.navigation.discrete import NavigationCommandKind as Kind
from embodirun.robots.unitree.go2.navigation.frames import EncodedRGBFrame
from embodirun.robots.unitree.go2.navigation.interfaces import MobileBaseError
from embodirun.robots.unitree.go2.navigation.motion import MobileBaseState
from embodirun.robots.unitree.go2.navigation.observation import NavigationObservation
from embodirun.utils import Pose2D

FORWARD = Command(Kind.MOVE_FORWARD, distance_m=0.1)  # .4 s on the fake base
STOP = Command(Kind.STOP)


class Base:
    def __init__(self):
        self.active = None
        self.expires = 0.0
        self.starts, self.updates, self.stops = [], [], []
        self.release = threading.Event()
        self.owner = threading.get_ident()
        self.renewal = True

    def state(self):
        if time.monotonic() >= self.expires:
            self.active = None
        return MobileBaseState(Pose2D(0, 0, 0), 0, 0, 0, 1, None, self.active)

    def preflight(self, *, require_renewal=False):
        if self.state().active_velocity_lease_id or (require_renewal and not self.renewal):
            raise MobileBaseError("base unavailable")
        return self.state()

    def start_velocity_lease(self, command, *, duration_s):
        assert threading.get_ident() == self.owner
        assert 0 < duration_s <= 10
        self.active = str(len(self.starts))
        self.starts.append(time.monotonic())
        self.expires = time.monotonic() + duration_s
        return self.active

    def update_velocity_lease(self, action_id, command, *, remaining_s=None):
        assert threading.get_ident() == self.owner
        assert not self.stops
        if self.state().active_velocity_lease_id != action_id:
            raise MobileBaseError("wrong lease")
        self.updates.append((time.monotonic(), command))
        if remaining_s is not None:
            assert 0.05 <= remaining_s <= 10
            self.expires = time.monotonic() + remaining_s

    def stop(self):
        assert threading.get_ident() == self.owner
        self.stops.append(time.monotonic())
        self.active = None
        self.release.set()


class Camera:
    def __init__(self, base):
        self.base = base
        self.captures = []
        self.metadata = True
        self.age = 0.0

    def capture(self, *, episode_id, reset, robot_state):
        self.captures.append((time.monotonic(), self.base.state().active_velocity_lease_id))
        seq = len(self.captures)
        return NavigationObservation(
            episode_id,
            seq,
            reset,
            (EncodedRGBFrame(seq, time.time(), b"\xff\xd8jpeg"),),
            metadata={"captured_at_monotonic_s": time.monotonic() - self.age, "capture_uncertainty_s": 0.001}
            if self.metadata
            else {},
        )


class Policy:
    def __init__(self, base):
        self.base = base
        self.calls, self.tails = [], []
        self.delay = 0.05
        self.error = None
        self.block = False
        self.entered = threading.Event()

    def plan(self, request):
        self.calls.append(threading.get_ident())
        return NavigationChunk(request.observation.sequence, (FORWARD,))

    def plan_after(self, request, remaining):
        self.calls.append(threading.get_ident())
        self.tails.append(remaining)
        self.entered.set()
        if self.block:
            assert self.base.release.wait(3), "worker must be drained AFTER stop"
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return NavigationChunk(request.observation.sequence, (FORWARD,) if len(self.tails) == 1 else (STOP,))


def setup_session(**kwargs):
    base = Base()
    camera = Camera(base)
    policy = Policy(base)
    events = []
    session = AsyncChunkNavigationSession(
        policy,
        camera,
        base,
        config=ChunkSessionConfig(max_runtime_s=5),
        overlap=OverlapConfig(lead_s=0.2),
        event_sink=events.append,
        **kwargs,
    )
    return session, base, camera, policy, events


def test_ready_continuations_overlap_capture_inference_and_control_without_zero():
    session, base, camera, policy, events = setup_session()
    result = session.run("go", execute=True)
    assert result["reason"] == "stop" and result["chunks"] == 3
    assert len(base.starts) == len(base.stops) == 1
    assert all(command.vx_mps > 0 for _, command in base.updates)
    assert len(policy.tails) == 2 and all(0 < tail[0].distance_m < FORWARD.distance_m for tail in policy.tails)
    assert any(active is not None for _, active in camera.captures)
    assert len(set(policy.calls)) == 1 and policy.calls[0] != base.owner
    assert [e["ready"] for e in events if e["event"] == "overlap_boundary"] == [True, True]
    assert result["capture_inference_overlap_s"] > 0.08


def test_late_inference_stops_at_known_tail_and_restarts_without_replaying():
    session, base, _, policy, _ = setup_session()
    policy.delay = 0.3
    result = session.run("go", execute=True)
    assert result["reason"] == "stop"
    assert len(base.starts) == 2
    assert sum(command.vx_mps == 0 for _, command in base.updates) == 2
    assert result["inference_wait_s"] > 0.05
    assert result["boundary_wait_s"] > 0.15


def test_inference_failure_stops_and_does_not_retry():
    session, base, _, policy, _ = setup_session()
    policy.error = ValueError("bad response")
    with pytest.raises(ValueError, match="bad response"):
        session.run("go", execute=True)
    assert len(base.stops) == 1 and len(policy.calls) == 2


def test_cancellation_stops_before_draining_inflight_model_and_never_restarts():
    session, base, _, policy, _ = setup_session()
    policy.block = True

    def sleep(seconds):
        if policy.entered.is_set():
            raise KeyboardInterrupt
        time.sleep(seconds)

    session.sleep = sleep
    with pytest.raises(KeyboardInterrupt):
        session.run("go", execute=True)
    assert len(base.stops) == 1 and len(base.starts) == 1
    assert not session._running
    assert not any(t.name.startswith("go2-observe-infer") for t in threading.enumerate())


@pytest.mark.parametrize("missing", [True, False])
def test_bad_frame_timing_fails_before_motion(missing):
    session, base, camera, _, _ = setup_session()
    camera.metadata = not missing
    camera.age = 10 if not missing else 0
    with pytest.raises((ValueError, TimeoutError)):
        session.run("go", execute=True)
    assert not base.starts and not base.stops


def test_stale_model_result_is_not_executed():
    session, base, _, policy, _ = setup_session()
    session.config = replace(session.config, max_plan_age_s=0.25)
    policy.delay = 0.35
    with pytest.raises(TimeoutError, match="too old"):
        session.run("go", execute=True)
    assert len(base.starts) == len(base.stops) == 1


def test_runtime_timeout_stops_before_draining():
    session, base, _, policy, _ = setup_session()
    session.config = replace(session.config, max_runtime_s=1.6)
    policy.block = True
    with pytest.raises(TimeoutError, match="runtime limit"):
        session.run("go", execute=True)
    assert len(base.stops) == 1


def test_remaining_tail_crosses_action_boundaries_and_keeps_partial_turns():
    chunk = NavigationChunk(
        1, (Command(Kind.MOVE_FORWARD, distance_m=0.25), Command(Kind.TURN_LEFT, angle_deg=30), FORWARD)
    )
    config = ChunkSessionConfig()
    tail = remaining_commands(chunk, 1.0 + (3.141592653589793 / 6) / config.yaw_rate_rps / 2, config)
    assert tail[0].angle_deg == pytest.approx(15)
    assert tail[1] == FORWARD
    assert remaining_commands(chunk, 100, config) == ()
