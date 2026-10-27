# Go2 / ActiveVLN on Orin NX

Use EmbodiRun `bench/activevln-navigation` (`d9f7efe`) together with EmbodiInfer
`fix/activevln-resident-kv` (`7ae18ec`). The entrypoint is
`embodirun go2 activevln`, using the existing execution-prefix-aware overlap
navigation and the Infer branch's resident-KV NX service.

See [the existing workflow](../../examples/README.md) and
[the local preparation/deployment commands](../zh/go2-activevln.md).
`configs/activevln/go2-orin-nx-onboard.json` fills the loopback endpoints for
inference (18767), camera (8765) and control (8080). `plan` is offline; `check`
is read-only; `run` without `--execute` performs one inference without motion.

The checkpoint revision remains `160987313e3e869705f42400d1b8f28177044518`.
The standalone download and BF16 conversion scripts preserve source weights
and write checksums. Preparation uses its own Python 3.10+ project and lock at
`scripts/activevln-preparation`; it installs neither Torch nor MicroDuck and
does not change the Host or inference environments. With uv 0.12.x, run these
commands from the repository root:

```bash
uv sync --project scripts/activevln-preparation --frozen
uv run --project scripts/activevln-preparation --frozen hf auth login
uv run --project scripts/activevln-preparation --frozen \
  python scripts/download_navigation_models.py \
  --repo Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
  --revision 160987313e3e869705f42400d1b8f28177044518 \
  --output ../models/activevln-r2r --workers 4 --authenticated
uv run --project scripts/activevln-preparation --frozen \
  python scripts/prepare_activevln_bf16.py \
  ../models/activevln-r2r ../models/activevln-r2r-bf16
```

Conversion requires verified checksums for the config and every indexed shard.
It excludes unverified files, preserves integer tensors and records nearest-even
BF16 conversion. A final `SHA256SUMS` is written only after complete conversion.
Run the artifact integrity and vendor startup regression tests without CUDA or
robot hardware:

```bash
uv run --project scripts/activevln-preparation --frozen --group test \
  python -m pytest tests/test_go2_preparation.py -q
```

The NX launcher also requires its pinned recorded R2R warmup assets and
256-episode selection manifest. These are not yet prepared locally. INT8 needs
an additional matching calibration file. The reference NX environment is L4T
R36.4.7 / CUDA 12.6, while the inspected robot uses R35.3.1 / CUDA 11.4; the
inference environment remains to be resolved. No GPU or physical-robot
validation is implied by local CPU tests.
