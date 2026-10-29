"""Routine task release must never re-arm or stop a different controller."""

import time
from dataclasses import replace

import pytest

from embodirun_go2_owner.control.executor import ActionExecutor
from embodirun_go2_owner.control.transport import DryRunTransport
from embodirun_go2_owner.control.types import ApiError

MOVE = {"action": "stream_move", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration_s": 5}


@pytest.fixture
def executor():
    owner = ActionExecutor(DryRunTransport(), operator_motion_ready=True)
    try:
        yield owner
    finally:
        owner.close()


def start(executor, **kwargs):
    return executor.execute({**MOVE, **kwargs})[1]["action_id"]


def release(executor, action_id):
    return executor.execute({"action": "release_move", "action_id": action_id})[1]


@pytest.mark.parametrize("completed", [False, True])
def test_release_allows_successive_tasks_without_rearming(executor, completed):
    for _ in range(2):
        action_id = start(executor, duration_s=0.1 if completed else 5)
        if completed:
            executor._worker.join(timeout=1)
            assert executor.snapshot()["last_result"]["status"] == "completed"
        result = release(executor, action_id)
        assert result["accepted"] and result["stopped"] and result["operator_motion_ready"]
        assert executor.snapshot()["active_action"] is None
        assert executor.transport.state().velocity == (0, 0, 0)
        assert release(executor, action_id)["stopped"]  # safe to retry this lease only


def test_stale_release_cannot_stop_new_owner(executor):
    old_id = start(executor)
    release(executor, old_id)
    new_id = start(executor)
    with pytest.raises(ApiError) as error:
        release(executor, old_id)
    assert error.value.status == 409
    assert executor.snapshot()["active_action"]["id"] == new_id
    assert not executor._stop_event.is_set()
    assert executor.operator_motion_ready


def test_release_after_emergency_stop_never_rearms(executor):
    action_id = start(executor)
    executor.stop()
    with pytest.raises(ApiError, match="interlock"):
        release(executor, action_id)
    assert not executor.operator_motion_ready
    with pytest.raises(ApiError, match="interlock"):
        start(executor)


@pytest.mark.parametrize("fault", ["stale", "tilt", "robot_error", "stop"])
def test_release_faults_revoke_readiness(executor, monkeypatch, fault):
    action_id = start(executor, duration_s=0.1)
    executor._worker.join(timeout=1)
    assert executor.snapshot()["last_result"]["status"] == "completed"
    if fault == "stop":
        monkeypatch.setattr(executor.transport, "stop", lambda: -1)
    else:
        state = executor.transport.state()
        changes = {
            "stale": {"received_at": time.monotonic() - 60},
            "tilt": {"roll": 3},
            "robot_error": {"error_code": 1},
        }
        monkeypatch.setattr(executor.transport, "state", lambda: replace(state, **changes[fault]))
    with pytest.raises(ApiError):
        release(executor, action_id)
    assert not executor.operator_motion_ready


def test_cancellation_does_not_mask_a_move_failure(executor, monkeypatch):
    def failed_move(*args):
        executor._stop_event.set()
        return 17

    monkeypatch.setattr(executor.transport, "move", failed_move)
    monkeypatch.setattr(executor.transport, "stop", lambda: 0)
    action_id = start(executor)
    executor._worker.join(timeout=1)
    assert executor.snapshot()["last_result"]["status"] == "failed"
    assert not executor.operator_motion_ready
    with pytest.raises(ApiError, match="safely"):
        release(executor, action_id)


def test_failed_worker_stop_cannot_be_hidden_by_successful_second_stop(executor, monkeypatch):
    action_id = start(executor)
    calls = []

    def stop():
        calls.append(1)
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(executor.transport, "stop", stop)
    with pytest.raises(ApiError, match="safely"):
        release(executor, action_id)
    assert len(calls) == 2
    assert executor.snapshot()["last_result"]["status"] == "failed"
    assert not executor.operator_motion_ready


@pytest.mark.parametrize("payload", [{"action": "release_move"}, {"action": "release_move", "action_id": " "}])
def test_release_requires_explicit_identity(executor, payload):
    action_id = start(executor)
    with pytest.raises(ApiError) as error:
        executor.execute(payload)
    assert error.value.status == 400
    assert executor.snapshot()["active_action"]["id"] == action_id


def test_release_cannot_cancel_posture_or_nonstream_action(executor):
    action_id = start(executor, action="move")
    with pytest.raises(ApiError) as error:
        release(executor, action_id)
    assert error.value.status == 409
    assert not executor._stop_event.is_set()
