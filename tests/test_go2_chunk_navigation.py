"""Whole-chunk execution tests with a clock and base that enforce lease ownership."""

from dataclasses import replace

import pytest

from embodirun.robots.unitree.go2.navigation.chunk_session import (
    ChunkNavigationSession,
    ChunkSessionConfig,
    NavigationChunk,
    velocity_segments,
)
from embodirun.robots.unitree.go2.navigation.discrete import NavigationCommand as Command
from embodirun.robots.unitree.go2.navigation.discrete import NavigationCommandKind as Kind
from embodirun.robots.unitree.go2.navigation.frames import EncodedRGBFrame
from embodirun.robots.unitree.go2.navigation.interfaces import MobileBaseError
from embodirun.robots.unitree.go2.navigation.motion import MobileBaseState
from embodirun.robots.unitree.go2.navigation.observation import NavigationObservation
from embodirun.utils import Pose2D

FORWARD = Command(Kind.MOVE_FORWARD, distance_m=0.25)
LEFT = Command(Kind.TURN_LEFT, angle_deg=15)
STOP = Command(Kind.STOP)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


class Base:
    def __init__(self, clock):
        self.clock = clock
        self.commands = []
        self.leases = []
        self.stops = 0
        self.lease_id = None
        self.expiry = 0
        self.armed = True

    def state(self):
        if self.clock() >= self.expiry:
            self.lease_id = None
        return MobileBaseState(Pose2D(0, 0, 0), 0, 0, 0, 1, None, self.lease_id)

    def preflight(self):
        state = self.state()
        if state.active_velocity_lease_id or not self.armed:
            raise MobileBaseError("not idle/armed")
        return state

    def start_velocity_lease(self, command, *, duration_s):
        self.preflight()
        assert duration_s <= 10
        self.lease_id = f"lease-{len(self.leases)}"
        self.leases.append((self.clock(), self.lease_id, duration_s))
        self.expiry = self.clock() + duration_s
        self.commands.append((self.clock(), self.lease_id, command))
        return self.lease_id

    def update_velocity_lease(self, lease_id, command):
        if self.state().active_velocity_lease_id != lease_id:
            raise MobileBaseError("expired lease")
        self.commands.append((self.clock(), lease_id, command))

    def stop(self):
        self.stops += 1
        self.lease_id = None
        self.armed = False


class Camera:
    def __init__(self, clock, base):
        self.clock, self.base = clock, base
        self.captures = []
        self.frozen = False

    def capture(self, *, episode_id, reset, robot_state):
        assert self.base.state().active_velocity_lease_id is None
        self.captures.append(self.clock())
        sequence = 1 if self.frozen else int(self.clock() * 5) + 1
        return NavigationObservation(
            episode_id,
            sequence,
            reset,
            (EncodedRGBFrame(sequence, 1000 + sequence / 5, b"\xff\xd8jpeg"),),
        )


class Policy:
    def __init__(self, clock, base, chunks):
        self.clock, self.base = clock, base
        self.chunks = iter(chunks)
        self.requests = []

    def plan(self, request):
        assert self.base.state().active_velocity_lease_id is None
        self.requests.append(request)
        self.clock.sleep(0.5)
        return NavigationChunk(request.observation.sequence, next(self.chunks))


def rig(chunks, **config):
    clock = Clock()
    base = Base(clock)
    camera = Camera(clock, base)
    policy = Policy(clock, base, chunks)
    session = ChunkNavigationSession(
        policy,
        camera,
        base,
        config=ChunkSessionConfig(**config),
        clock=clock,
        sleep=clock.sleep,
    )
    return session, clock, base, camera, policy


def test_full_chunk_uses_one_lease_and_has_no_stop_at_action_seams():
    session, _, base, camera, policy = rig([(FORWARD, FORWARD, LEFT), (STOP,)])
    result = session.run("go to the door", execute=True)
    assert result["reason"] == "stop"
    assert result["executed_actions"] == 4
    assert len(base.leases) == 1 and base.stops == 1
    velocities = [command for _, _, command in base.commands]
    assert all(v.moving for v in velocities[:-1])
    assert not velocities[-1].moving
    assert all(lease == base.leases[0][1] for _, lease, _ in base.commands)
    start = base.leases[0][0]
    turn = next(t for t, _, command in base.commands if command.yaw_rate_rps > 0)
    assert turn - start == pytest.approx(2.0)
    assert policy.requests[0].observation.reset is True
    assert policy.requests[1].observation.reset is False
    assert (
        policy.requests[1].observation.latest_rgb.captured_at_s
        > policy.requests[0].observation.latest_rgb.captured_at_s
    )
    assert len(camera.captures) > 2  # Waits for a new post-motion frame.


def test_merging_preserves_distance_and_turn_sign():
    config = ChunkSessionConfig()
    forward = velocity_segments(NavigationChunk(1, (FORWARD, FORWARD, FORWARD)), config)
    assert len(forward) == 1
    assert forward[0].duration_s * forward[0].command.vx_mps == pytest.approx(0.75)
    right = velocity_segments(NavigationChunk(1, (Command(Kind.TURN_RIGHT, angle_deg=45), STOP)), config)
    assert right[0].command.yaw_rate_rps < 0
    assert len(right) == 1


def test_next_chunk_gets_a_fresh_lease_without_disarming_between_chunks():
    session, _, base, _, _ = rig([(FORWARD,), (FORWARD, STOP)])
    assert session.run("go", execute=True)["reason"] == "stop"
    assert len(base.leases) == 2 and base.stops == 1


@pytest.mark.parametrize("chunk", [(FORWARD, FORWARD, LEFT), (STOP,)])
def test_dry_run_has_no_control_writes_even_stop(chunk):
    session, _, base, _, policy = rig([chunk])
    base.armed = False  # Dry-run must not require arming.
    assert session.run("go")["reason"] == "dry_run"
    assert len(policy.requests) == 1
    assert base.commands == [] and base.stops == 0


def test_immediate_stop_has_no_motion():
    session, _, base, _, _ = rig([(STOP,)])
    assert session.run("go", execute=True)["reason"] == "stop"
    assert base.commands == []


def test_oversized_chunk_is_rejected_whole():
    session, _, base, _, _ = rig([(FORWARD, FORWARD, FORWARD)], max_chunk_s=2)
    with pytest.raises(ValueError, match="refusing to truncate"):
        session.run("go", execute=True)
    assert not base.leases and base.stops == 0


def test_frozen_camera_never_reaches_model_or_motion():
    session, _, base, camera, policy = rig([(FORWARD,)])
    camera.frozen = True
    with pytest.raises(TimeoutError, match="new post-motion frame"):
        session.run("go", execute=True)
    assert not policy.requests and not base.commands and base.stops == 0


def test_stale_inference_is_not_executed():
    session, _, base, _, _ = rig([(FORWARD,)], max_plan_age_s=0.2)
    with pytest.raises(TimeoutError, match="too old"):
        session.run("go", execute=True)
    assert not base.commands


@pytest.mark.parametrize("failure", [MobileBaseError("lost transport"), KeyboardInterrupt()])
def test_write_failure_and_cancellation_stop_without_retry(failure):
    session, _, base, _, policy = rig([(FORWARD,)])

    def fail(*args):
        raise failure

    base.update_velocity_lease = fail
    with pytest.raises(type(failure)):
        session.run("go", execute=True)
    assert len(base.leases) == 1 and base.stops == 1 and len(policy.requests) == 1


def test_ambiguous_start_failure_still_stops():
    session, _, base, _, _ = rig([(FORWARD,)])
    original = base.start_velocity_lease

    def fail(command, *, duration_s):
        original(command, duration_s=duration_s)
        raise MobileBaseError("response lost after acceptance")

    base.start_velocity_lease = fail
    with pytest.raises(MobileBaseError):
        session.run("go", execute=True)
    assert base.stops == 1


def test_control_stall_stops_instead_of_catching_up():
    session, clock, base, _, _ = rig([(FORWARD,)])
    original_sleep = clock.sleep

    def delayed(seconds):
        original_sleep(seconds + (0.8 if base.lease_id else 0))

    session.sleep = delayed
    with pytest.raises(TimeoutError, match="heartbeat delayed"):
        session.run("go", execute=True)
    assert base.stops == 1 and len(base.leases) == 1


def test_rejected_preflight_does_not_stop_another_owner():
    session, _, base, _, _ = rig([(FORWARD,)])
    base.lease_id, base.expiry = "someone-else", 100
    with pytest.raises(MobileBaseError):
        session.run("go", execute=True)
    assert not base.commands and base.stops == 0


def test_delay_cannot_silently_skip_a_short_turn_within_a_chunk():
    session, clock, base, _, _ = rig([(FORWARD, LEFT, FORWARD)], yaw_rate_rps=0.7)

    def delayed(seconds):
        if base.leases and 0.85 < clock() - base.leases[0][0] < 1.0:
            seconds = 0.49  # Below max_control_gap_s but spans the entire turn.
        clock.sleep(seconds)

    session.sleep = delayed
    with pytest.raises(TimeoutError, match="skipped an action"):
        session.run("go", execute=True)
    assert base.stops == 1


def test_runtime_limit_does_not_start_a_chunk_it_cannot_finish():
    session, _, base, _, _ = rig([(FORWARD,)], max_runtime_s=1)
    assert session.run("go", execute=True)["reason"] == "max_runtime"
    assert not base.commands


def test_another_owner_after_our_completed_chunk_is_not_stopped():
    session, _, base, _, policy = rig([(FORWARD,), (FORWARD,)])
    original = policy.plan

    def takeover(request):
        chunk = original(request)
        if len(policy.requests) == 2:
            base.lease_id, base.expiry = "other-owner", 100
        return chunk

    policy.plan = takeover
    with pytest.raises(MobileBaseError):
        session.run("go", execute=True)
    assert len(base.leases) == 1 and base.stops == 0
    assert base.lease_id == "other-owner"


def test_wrong_observation_identity_aborts_before_motion():
    session, _, base, _, policy = rig([(FORWARD,)])
    original = policy.plan
    policy.plan = lambda request: replace(original(request), observation_sequence=999)
    with pytest.raises(ValueError, match="does not match"):
        session.run("go", execute=True)
    assert not base.commands


@pytest.mark.parametrize(
    "values",
    [
        {"control_hz": 1},
        {"control_hz": 50},
        {"max_chunks": True},
        {"max_chunks": 1.5},
        {"linear_speed_mps": 0.36},
        {"yaw_rate_rps": 0.71},
        {"max_chunk_s": 10},
        {"max_control_gap_s": 2},
        {"max_plan_age_s": float("nan")},
    ],
)
def test_invalid_configuration_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        ChunkSessionConfig(**values)
