"""Audit completed NX navigation traces and summarize paired quality and model timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from metrics import compare, primitive_actions


def read(path: Path) -> Any:
    """Read one retained JSON evidence artifact."""
    return json.loads(path.read_text())


def timing_summary(values: list[float]) -> dict[str, float]:
    """Use linear percentiles over measured calls, without including warmup."""
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError("timings must be nonempty, finite and nonnegative")

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {"mean": statistics.mean(ordered), "p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


def summarize_stage(root: Path, stage: str) -> dict[str, Any]:
    """Require all declared episodes and verify actions, distances, timings and model-call evidence."""
    split = stage.rsplit("-", 1)[1]
    folder = root / stage
    report = read(folder / f"{split}.json")
    count = report.get("expected_episodes", 48)
    if report["status"] != "complete" or len(report["episodes"]) != count:
        raise ValueError(f"{stage} is incomplete; expected {count} episodes")
    if len({str(ep["episode_id"]) for ep in report["episodes"]}) != count:
        raise ValueError("duplicate completed episode IDs")
    details, timings, http, forward = [], [], [], []
    for summary in report["episodes"]:
        episode = read(folder / split / f"episode-{summary['episode_id']}.json")
        for key in (
            "episode_id",
            "status",
            "success",
            "spl",
            "steps",
            "stopped",
            "path_length_m",
            "distance_to_goal_m",
        ):
            if summary[key] != episode[key]:
                raise ValueError(f"summary/trace mismatch: {key}")
        position, steps, length, stopped = episode["start_position"], 0, 0.0, False
        distances = [episode["shortest_path_m"]]
        for trace in episode["trace"]:
            if trace["before_position"] != position:
                raise ValueError("discontinuous execution trace")
            timings.append(trace["timing"]["e2e_ms"])
            forward.append(trace["timing"]["pure_inference_ms"])
            http.append(trace["http_roundtrip_ms"])
            if "invalid_action" in trace:
                if episode["termination"] != "invalid_action":
                    raise ValueError("invalid action was not reported as such")
                continue
            expected = primitive_actions(trace["model"]["rows"])[: 500 - steps]
            if [row["action"] for row in trace["executed"]] != expected:
                raise ValueError("executed primitives differ from model actions")
            for row in trace["executed"]:
                if stopped:
                    raise ValueError("action executed after STOP")
                length += math.dist(position, row["position"])
                position, stopped = row["position"], row["action"] == 0
                steps += 1
                if row["distance_to_goal_m"] is not None:
                    distances.append(row["distance_to_goal_m"])
        if steps != episode["steps"] or stopped != episode["stopped"] or abs(length - episode["path_length_m"]) > 1e-4:
            raise ValueError("primitive trajectory does not reproduce episode metrics")
        episode["oracle_success"] = min(distances) < 3.0
        details.append(episode)
    # Also recompute STOP/SPL evidence with the independent metric validator.
    compare(report, report)
    evidence = [json.loads(line) for line in (folder / "tensor-batches.jsonl").read_text().splitlines()]
    if len(evidence) != len(timings) or any(
        row["status"] != "complete" or row["observations"] != 1 for row in evidence
    ):
        raise ValueError("server calls differ from retained batch-one trajectories")
    starts = [row["rows"][0] for row in evidence if row["rows"][0]["step_id"] == 0]
    if len(starts) != count or any(row["cache_length"] != 0 for row in starts):
        raise ValueError("episode histories were not reset independently")
    distances = [ep["distance_to_goal_m"] for ep in details if ep["distance_to_goal_m"] is not None]
    return {
        "stage": stage,
        "status": "complete",
        "episodes": count,
        "successes": sum(ep["success"] for ep in details),
        "sr": statistics.mean(ep["success"] for ep in details),
        "spl": statistics.mean(ep["spl"] for ep in details),
        "navigation_error_m": statistics.mean(distances) if len(distances) == count else None,
        "unreachable_final_positions": count - len(distances),
        "oracle_sr": statistics.mean(ep["oracle_success"] for ep in details),
        "mean_path_length_m": statistics.mean(ep["path_length_m"] for ep in details),
        "mean_primitive_steps": statistics.mean(ep["steps"] for ep in details),
        "invalid_action_episode_ids": [ep["episode_id"] for ep in details if ep["termination"] == "invalid_action"],
        "model_calls": len(timings),
        "model_e2e_ms": timing_summary(timings),
        "model_forward_ms": timing_summary(forward),
        "http_roundtrip_ms": timing_summary(http),
        "model_inference_seconds": sum(timings) / 1000,
        "episode_wall_seconds": sum(ep["wall_seconds"] for ep in details),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in evidence),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in evidence),
        "runtime": evidence[-1]["runtime"],
        "manifest_sha256": report["manifest_sha256"],
        "report_sha256": hashlib.sha256((folder / f"{split}.json").read_bytes()).hexdigest(),
        "selection": report.get("selection"),
        "protocol": report["protocol"],
        "model_service": report["model_service"],
        "calibration_overlap_episode_ids": report.get("calibration_overlap_episode_ids", []),
        "timing_scope": "Decoded CPU RGB to CPU actions; excludes disk, warmup, HTTP transport and simulator execution. Closed-loop trajectories and call counts can differ.",
    }


def paired_summary(root: Path, datasets: tuple[str, ...] = ("r2r", "rxr")) -> dict[str, Any]:
    """Join complete pairs, requiring identical sources, settings and initial observations."""
    result: dict[str, Any] = {"acceptance_threshold": None, "datasets": {}}
    if not datasets or any(split not in ("r2r", "rxr") for split in datasets) or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be a nonempty unique selection of r2r/rxr")
    for split in datasets:
        stages = [f"{precision}-{split}" for precision in ("bf16", "int8")]
        reports = [read(root / stage / f"{split}.json") for stage in stages]
        pair = compare(*reports, max_fewer=0)
        for key in ("max_fewer_successes", "success_rate_within_tolerance", "latency_admission_evaluated"):
            pair.pop(key)
        ready = [r["model_service"] for r in reports]
        for key in ("source_python_sha256",):
            if ready[0]["environment"][key] != ready[1]["environment"][key]:
                raise ValueError("paired model sources differ")
        if ready[0]["allocator_config"] != ready[1]["allocator_config"]:
            raise ValueError("paired allocators differ")
        for report, service in zip(reports, ready, strict=True):
            if service.get("navigation_manifest_sha256", report["manifest_sha256"]) != report["manifest_sha256"]:
                raise ValueError("service prewarm and navigation episode selections differ")
        for key in ("source_snapshot", "adapter_sha256", "loading_adapter_sha256", "float32_matmul_precision"):
            if ready[0].get(key) != ready[1].get(key):
                raise ValueError(f"paired services differ: {key}")
        # Output paths are run-local; the manifest's contents are compared above.
        ignored = {"precision_experiment", "output_dir", "navigation_manifest"}
        configs = [{k: v for k, v in r["config"].items() if k not in ignored} for r in ready]
        if configs[0] != configs[1]:
            raise ValueError("paired runtime configurations differ")
        for summary in reports[0]["episodes"]:
            episodes = [read(root / stage / split / f"episode-{summary['episode_id']}.json") for stage in stages]
            for key in ("start_position", "start_rotation", "shortest_path_m", "habitat_sim_version"):
                if episodes[0][key] != episodes[1][key]:
                    raise ValueError(f"paired initial condition differs: {key}")
            if episodes[0]["trace"][0]["image_sha256"] != episodes[1]["trace"][0]["image_sha256"]:
                raise ValueError("paired initial RGB differs")
        pair["profiles"] = {stage.split("-")[0]: read(root / stage / "summary.json") for stage in stages}
        pair["paired_initial_observation_audits"] = len(reports[0]["episodes"])
        result["datasets"][split] = pair
    return result


def main() -> None:
    """Summarize one complete stage or compare the two complete dataset pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("bf16-r2r", "int8-r2r", "bf16-rxr", "int8-rxr", "compare"))
    parser.add_argument("--datasets", choices=("r2r", "rxr"), nargs="+", default=["r2r", "rxr"])
    args = parser.parse_args()
    result = (
        paired_summary(args.root, tuple(args.datasets))
        if args.stage == "compare"
        else summarize_stage(args.root, args.stage)
    )
    output = args.root / "comparison.json" if args.stage == "compare" else args.root / args.stage / "summary.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "status": "complete", "episodes": result.get("episodes")}))


if __name__ == "__main__":
    main()
