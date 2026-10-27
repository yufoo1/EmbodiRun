"""Lease renewal extends only live bounded streams, retaining watchdog behavior."""

import threading
from types import SimpleNamespace

import pytest

from embodirun.robots.unitree.go2.agent.control.admission import MotionAdmission, admit_motion_update
from embodirun.robots.unitree.go2.agent.control.lease import MotionLease
from embodirun.robots.unitree.go2.agent.control.types import ApiError


def update(lease, remaining=0.3):
    return admit_motion_update({"action_id": lease.id, "vx": 0.1, "vy": 0.0, "yaw_rate": 0.0, "remaining_s": remaining})


def new_lease(action="stream_move", duration=0.15):
    return MotionLease(MotionAdmission(action, {"vx": 0.1, "vy": 0.0, "yaw_rate": 0.0, "duration_s": duration}))


@pytest.mark.parametrize("remaining", [True, -1, 0, 0.01, 10.01, float("nan"), float("inf")])
def test_renewal_admission_remains_bounded(remaining):
    with pytest.raises(ApiError):
        update(new_lease(), remaining)


@pytest.mark.parametrize("cause", ["deadline", "heartbeat", "stop", "finished", "timed_move"])
def test_expired_cancelled_or_nonstream_lease_cannot_be_renewed(cause):
    lease = new_lease("move" if cause == "timed_move" else "stream_move")
    if cause == "deadline":
        lease._deadline = 0.0
    elif cause == "heartbeat":
        lease.heartbeat_at_monotonic = 0.0
    elif cause == "stop":
        lease.stop_event.set()
    elif cause == "finished":
        lease._finished = True
    with pytest.raises(ApiError):
        lease.update(update(lease))


def test_stream_extends_across_original_deadline_without_stop_or_watchdog_bypass(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("embodirun.robots.unitree.go2.agent.control.lease.time.monotonic", lambda: now[0])
    lease = new_lease()
    stopped = []
    moved = []

    def move(*args):
        moved.append(now[0])
        if len(moved) == 3:
            lease.update(update(lease))
        return 0

    monkeypatch.setattr(lease.stop_event, "wait", lambda duration: now.__setitem__(0, now[0] + duration))
    transport = SimpleNamespace(move=move, stop=lambda: stopped.append(now[0]) or 0)
    result = lease.run(transport, threading.Lock(), lambda: SimpleNamespace(yaw=0))
    assert result["status"] == "completed"
    assert max(moved) > 0.15
    assert len(stopped) == 1 and 0.4 <= stopped[0] <= 0.45
    with pytest.raises(ApiError):
        lease.update(update(lease))


def test_renewal_does_not_disable_heartbeat_expiry(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("embodirun.robots.unitree.go2.agent.control.lease.time.monotonic", lambda: now[0])
    lease = new_lease(duration=9)
    lease.update(update(lease, 10))
    monkeypatch.setattr(lease.stop_event, "wait", lambda duration: now.__setitem__(0, now[0] + duration))
    stopped = []
    result = lease.run(
        SimpleNamespace(move=lambda *args: 0, stop=lambda: stopped.append(now[0]) or 0),
        threading.Lock(),
        lambda: SimpleNamespace(yaw=0),
    )
    assert result["status"] == "failed" and result["detail"] == "stream heartbeat expired"
    assert len(stopped) == 1 and 2 <= stopped[0] < 2.1
