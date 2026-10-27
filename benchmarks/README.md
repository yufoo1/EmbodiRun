# Benchmarks

Measure inference, transport, and runtime behavior with defined workloads and
machine-readable results. For robot and simulator task setup, use
[Recipes](../examples/README.md).

## Available measurements

| Benchmark | Workload | Results |
|---|---|---|
| [SO-101 engine comparison](engine-comparison/README.md) | π0.5 grasping with EmbodiInfer HTTP/WirelessComm, SGLang, and native LeRobot | [Timing report](../docs/en/demos/engine-e2e-contrast.md), summary JSON, and SVG generator |
| [Inference transport](inference-transport/README.md) | HTTP / WirelessComm replay against the same model service and recorded observations | [Transport report](../docs/en/inference-transport.md) and measurement commands |
| [ActiveVLN navigation](#activevln-navigation) | Closed-loop Habitat R2R/RxR evaluation through an EmbodiInfer HTTP service | [Orin NX R2R BF16 / INT8 results](https://github.com/yufoo1/EmbodiInfer/tree/fix/activevln-resident-kv/benchmarks/activevln-benchmark/orin-nx) |

Model-only benchmarks are maintained in
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer/tree/main/benchmarks).
中文结果：[引擎对比](../docs/zh/demos/engine-e2e-contrast.md) ·
[推理传输](../docs/zh/inference-transport.md)。

## ActiveVLN navigation

[activevln-navigation/](activevln-navigation/) runs closed-loop Habitat evaluation
against an EmbodiInfer HTTP service. The published Orin NX comparison uses R2R
val-unseen, 256 identical scene-proportional episodes per precision, batch one.
Results and inference configuration live in the
[Orin NX benchmark](https://github.com/yufoo1/EmbodiInfer/tree/fix/activevln-resident-kv/benchmarks/activevln-benchmark/orin-nx).
The runner uses the current EmbodiInfer HTTP client and negotiates the schema
advertised by frozen inference services. Habitat-Sim 0.2.4 runs separately.

Start the NX service using its benchmark instructions and forward its port to
the simulator host. Copy its `config.json`, `service-ready.json`, `service-token`
and `selected-r2r-navigation-256.json` into `RUN_ROOT`. From this repository:

```bash
# Set RUN_ROOT, HABITAT_PYTHON, SCENES and RESULTS to prepared local paths.
STAGE=bf16-r2r  # repeat with the INT8 service and STAGE=int8-r2r
PYTHONPATH=src python benchmarks/activevln-navigation/benchmark.py \
  --root "$RUN_ROOT" --config "$RUN_ROOT/config.json" \
  --endpoint http://127.0.0.1:18767 --service-ready "$RUN_ROOT/service-ready.json" \
  --token-file "$RUN_ROOT/service-token" --habitat-python "$HABITAT_PYTHON" \
  --scenes "$SCENES" --simulator-gpu 0 --datasets r2r --episode-count 256 \
  --batch-size 1 --output "$RESULTS/$STAGE"

# Copy the matching NX tensor-batches.jsonl to "$RESULTS/$STAGE/", then:
python benchmarks/activevln-navigation/summarize.py "$RESULTS" "$STAGE"
# After both stages:
python benchmarks/activevln-navigation/summarize.py "$RESULTS" compare --datasets r2r
```

Reports retain SR, SPL, NE, oracle SR, per-call model/HTTP timing, episode wall
time, CUDA peaks and action traces. Model E2E excludes warmup, HTTP and Habitat;
interrupted runs are incomplete, not navigation failures. Pair comparison checks
initial observations, episode selection, inference source and runtime settings.

Use the retained manifests for the published results; inference verifies their
SHA256. `prepare_selection.py` can rebuild them from the original datasets and
checked 1024-episode reachability manifests (`--preflight-root`), or perform new
reachability checks with the Habitat interpreter when that option is omitted:

```bash
"$HABITAT_PYTHON" benchmarks/activevln-navigation/prepare_selection.py \
  --datasets r2r --r2r-source "$R2R_SOURCE" --scenes "$SCENES" \
  --preflight-root "$PREFLIGHT_ROOT" --episode-count 256 --seed 42 --output "$MANIFESTS"
```

Weights, scenes, manifests, credentials and raw reports stay outside Git.
RxR is supported by the scripts but has no completed 256-episode result here.

## Run a measurement

The [transport guide](inference-transport/README.md) covers input preparation,
measurement, and report comparison:

- `benchmark.py run` measures a replay workload; `compare` checks comparability
  and prints results. `fixture` creates synthetic smoke-check inputs.
- `capture_observations.py` records camera and state inputs from a prepared
  device without commanding motion.
- `profile_inference.py` serves and analyzes inference-stage timings.

Working reports are written under ignored `artifacts/` directories.
Publishable results and figure generators belong beside their benchmark.

## Measurement contract

A result should include the commands, source samples, and configuration needed
to reproduce it:

| Record | Contents |
|---|---|
| Samples | JSON/JSONL or CSV timings, units, trial/client identifiers, completion status, and failures |
| Environment | Hardware, network topology, source revisions, checkpoint/input hashes, dtype, and engine flags |
| Workload | Warmup, measured iterations, concurrency, batch limit, collection window, action horizon, and seed |
| Report | Sample count, P50/P95 latency, completed-request throughput, failure rate, and variation across repeated runs |
| Figure | Deterministic generation command and links to the input data |

Specify timing boundaries for each metric. Keep credentials and private
calibration data local, and record unavailable metadata explicitly.

## Deployment work procedure

[Deployment and Runtime work](deployment-runtime/README.md) supplies six paired
procedures, an existing-command software service comparison, and empty CSV,
environment and report templates. An Agent has executed
[three prepared Control startup pairs and a separate port fault/recovery](deployment-runtime/results/2026-10-08-thor-software/report.md)
on Thor; these coarse software samples do not establish a performance benefit.
It separates direct
versus Host management of the same simulated Control service from the full
same-model LeRobot comparison, which still needs assets and a compatible adapter.
中文职责说明：[Runtime 的职责与部署成本](../docs/zh/runtime-value.md)。

## Planned benchmarks

| Benchmark | Workload | Status |
|---|---|---|
| [Deployment and Runtime work](deployment-runtime/README.md) | Same-service preparation, readiness, updates and diagnosis; six full-policy deployment costs | [Prepared software lifecycle samples](deployment-runtime/results/2026-10-08-thor-software/report.md) available; full same-model LeRobot costs remain pending |
| Client scaling | 1, 2, 4, and 8 clients sharing one service; `--max-batch 1` and opt-in π0.5 batching | 🟨 待补充 |
| Physical-node scaling | 1, 2, and 4 inference hosts with a fixed checkpoint and recorded client placement | 🟨 待补充 |
| Archived transport results | Public raw replay reports and a table/figure generator | 🟨 待补充 |

For deployment comparisons, separate environment preparation, model loading,
compilation/warmup, and control initialization. Use the same readiness checks
for every comparison route. Measure Runtime overhead with identical observations,
inference endpoints, action horizons, and a simulated executor.

Scaling sweeps should report aggregate and per-client throughput, P50/P95
latency, queueing, failures, GPU utilization, and memory. Record `--max-wait-ms`
and keep checkpoint, dtype, decode steps, and observation inputs fixed. Report
replay and action-playback workloads separately, since playback can limit
closed-loop throughput.
