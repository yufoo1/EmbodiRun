# Go2 + ActiveVLN：Orin NX 本体推理

使用这两条开发分支：

- EmbodiRun：`bench/activevln-navigation`，本次核对提交 `d9f7efe`。
- EmbodiInfer 子模块：`fix/activevln-resident-kv`，本次核对提交 `7ae18ec`。

运行入口是 **`embodirun go2 activevln`**，使用该分支已有的异步导航和 resident-KV
推理服务。`configs/activevln/go2-orin-nx-onboard.json` 已将三项服务地址设为 Orin
本机回环地址；推理端口为 18767，相机 8765，控制 8080。默认 `mode=overlap`，
可改为 `chunk` 做同步对照。完整行为说明见 [examples](../../examples/README.md)。

## Mac：下载与精度转换

在仓库根目录使用 uv 0.12.x。`scripts/activevln-preparation` 是独立的 Python 3.10+
准备环境，有自己的依赖锁文件，不安装 Torch 或 MicroDuck，也不改变 Host/模型环境。
以下命令均从仓库根目录执行：

```bash
uv sync --project scripts/activevln-preparation --frozen
uv run --project scripts/activevln-preparation --frozen hf auth login
uv run --project scripts/activevln-preparation --frozen \
  python scripts/download_navigation_models.py \
  --repo Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
  --revision 160987313e3e869705f42400d1b8f28177044518 \
  --output ../models/activevln-r2r --workers 4 --authenticated

# 等下载生成 SHA256SUMS 后再转换；原始文件保持不变。
uv run --project scripts/activevln-preparation --frozen \
  python scripts/prepare_activevln_bf16.py \
  ../models/activevln-r2r ../models/activevln-r2r-bf16
```

该 revision 与 NX benchmark 配置一致。FP32 原始快照约 16.28 GB，BF16 副本约
8.14 GB。下载逐文件核对上游哈希；转换要求配置和索引中的每个分片均通过校验，
使用 nearest-even，并只为完整输出生成独立校验文件。额外的未校验文件不会带入副本。
这不等同于已经在设备上验证模型输出。HF 凭证只保存在标准凭证目录。

准备工具和独立 Go2 agent 启动入口的回归测试可在这个环境执行，无需 CUDA 或硬件：

```bash
uv run --project scripts/activevln-preparation --frozen --group test \
  python -m pytest tests/test_go2_preparation.py -q
```

## 网线传输

本次确认的 Orin 地址是 `192.168.123.28`，SSH 用户为 `unitree`。在 Mac 执行：

```bash
rsync -av --partial --progress \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  ./ unitree@192.168.123.28:/home/unitree/EmbodiRun/
rsync -av --partial --progress ../models/activevln-r2r-bf16/ \
  unitree@192.168.123.28:/home/unitree/models/activevln-r2r-bf16/
```

在 Orin 的权重目录执行 `sha256sum -c SHA256SUMS`。Mac 的 `.venv` 不用于 Linux ARM64。

## Orin：运行前提

当前设备检查记录是 L4T R35.3.1、CUDA 11.4、Python 3.8、NX 16GB；分支的
[NX 参考配置](../../third_party/embodiinfer/benchmarks/activevln-benchmark/orin-nx/config.json)
记录的是 **L4T R36.4.7、CUDA 12.6、Python 3.10.20、Torch 2.10.0、Triton 3.4.0**。
两者不匹配，仍需核验设备上已有的独立环境/容器及可行部署方式。不要把普通 CUDA
13.0 锁文件当成 Jetson 安装方案；本地准备不包含系统升级或刷机。

用厂商 Python 读取相机序列号、深度单位和可用 profile，不启动视频流：

```bash
cd /home/unitree/EmbodiRun
/usr/bin/python3 scripts/inspect_go2_onboard.py --camera
```

输出的 `launch_command` 启动 640x480@15 RGB-D 相机 agent。模型环境确定后，用该
环境的 Python 执行 `scripts/inspect_go2_onboard.py --torch` 检查版本和 CUDA。

控制 agent 可先以未启用运动的状态启动，检查 SDK2 和状态订阅：

```bash
PYTHONPATH="$PWD/src/embodirun/robots/unitree:$HOME/unitree_sdk2_python" /usr/bin/python3 \
  -m go2.agent.control --mode live --host 127.0.0.1 --port 8080 --interface eth0 \
  --cyclonedds-lib-dir /home/unitree/cyclonedds/install/lib --no-operator-ready
```

单独暴露 `go2` 包可避免厂商 Python 3.8 导入要求 Python 3.10+ 的 Host 父包。
相机和控制代码都必须使用当前 Run 分支，overlap 需要帧时间和运动租约续期支持。

## 启动 resident-KV 推理服务

使用 [Infer 的 NX 启动器](../../third_party/embodiinfer/benchmarks/activevln-benchmark/orin-nx/README.md)，
从 BF16 开始。该启动器需要其固定的 R2R warmup 数据和 256 集选择 manifest，
不仅是模型权重；这些文件目前尚未在此 Mac 准备齐全。Orin 不需要启动 Habitat。

所需参数为：

```bash
# INFER_PY：已验证的 NX 推理 Python；WARMUP_R2R / MANIFEST：固定的预热资产。
"$INFER_PY" third_party/embodiinfer/benchmarks/activevln-benchmark/orin-nx/run.py bf16 \
  --dataset r2r --checkpoint /home/unitree/models/activevln-r2r-bf16 \
  --data-root "$WARMUP_R2R" --manifest "$MANIFEST" \
  --output "$MODEL_RUN" --port 18767
```

`MODEL_RUN` 必须是新的输出目录；等其 `service-ready.json` 生成后再连接。
启动器校验预热资产和 manifest 哈希。INT8 另需匹配的校准文件，不在本次默认路径内。

## 检查和运行

Host 使用单独的 Python 3.10+ 环境安装 EmbodiRun。服务均启动后，在 Orin 执行：

```bash
CONFIG=configs/activevln/go2-orin-nx-onboard.json
export ACTIVEVLN_BENCHMARK_TOKEN="$(cat "$MODEL_RUN/service-token")"
embodirun go2 activevln plan --config "$CONFIG"
embodirun go2 activevln check --config "$CONFIG"
embodirun go2 activevln run --config "$CONFIG" \
  --instruction "Go to the door and stop."
```

`plan` 完全离线；`check` 只读服务；上面的 `run` 只推理一次，不写运动命令。
真实运行前，操作员确认场地和停止手段，将控制 agent 的 `--no-operator-ready`
改为 `--operator-ready` 后重新启动，再给 `run` 添加 `--execute`。
任务结束会停止并撤销运动许可；下一次真机任务需重新确认就绪。

本地验证仅覆盖软件路径；尚未在这台 Go2/NX 上验证 GPU 推理、相机和运动。
overlap 使用运动中的图像和已承诺动作前缀，其实际效果仍需真机测试；深度不是避障控制器。
