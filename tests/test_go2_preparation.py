"""Artifact integrity and vendor-agent startup without CUDA or hardware."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_checksums(source):
    rows = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(source).as_posix()}\n")
    (source / "SHA256SUMS").write_text("".join(rows))


@pytest.fixture
def checkpoint(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ml_dtypes")
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    source = tmp_path / "fp32"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "torch_dtype": "float32",
                "vision_config": {"torch_dtype": "float32"},
            }
        )
    )
    save_file(
        {"weights": np.array([1.00390625, 1.01171875, -1.00390625], dtype=np.float32)},
        source / "model-00001.safetensors",
        metadata={"format": "pt"},
    )
    save_file({"ids": np.array([1, 2], dtype=np.int64)}, source / "model-00002.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 28},
                "weight_map": {"weights": "model-00001.safetensors", "ids": "model-00002.safetensors"},
            }
        )
    )
    (source / "tokenizer.json").write_text('{"version": "1.0"}')
    write_checksums(source)
    return source


def test_bf16_conversion_rounding_shard_index_hashes_and_source_preservation(checkpoint, tmp_path):
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import load_file

    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    # Stray transfer artifacts must not enter the verified output snapshot.
    (checkpoint / "unverified.json").write_text("not part of the snapshot")
    (checkpoint / "model-00003.safetensors.partial").write_bytes(b"incomplete")
    module = load_script("prepare_activevln_bf16")
    destination = tmp_path / "bf16"
    module.convert(checkpoint, destination)
    tensor = load_file(destination / "model-00001.safetensors")["weights"]
    assert tensor.dtype == ml_dtypes.bfloat16
    assert tensor.astype(np.float32).tolist() == [1, 1.015625, -1]
    assert load_file(destination / "model-00002.safetensors")["ids"].tolist() == [1, 2]
    with safe_open(destination / "model-00001.safetensors", framework="numpy") as reader:
        assert reader.metadata() == {"format": "pt"}
    assert all((checkpoint / name).read_bytes() == content for name, content in before.items())
    assert (destination / "tokenizer.json").read_bytes() == before["tokenizer.json"]
    assert not (destination / "unverified.json").exists()
    assert not list(destination.glob("*.partial"))
    config = json.loads((destination / "config.json").read_text())
    assert config["torch_dtype"] == config["vision_config"]["torch_dtype"] == "bfloat16"
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 22  # 3 BF16 values + 2 int64 values.
    assert index["weight_map"] == json.loads(before["model.safetensors.index.json"])["weight_map"]
    entries = [line.split("  ", 1) for line in (destination / "SHA256SUMS").read_text().splitlines()]
    assert {name for _, name in entries} == {path.name for path in destination.iterdir()} - {"SHA256SUMS"}
    for expected, name in entries:
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("failure", ["missing_manifest", "missing_config_hash", "missing_shard_hash", "corrupt_shard"])
def test_incomplete_or_corrupt_checkpoint_never_publishes_conversion(checkpoint, tmp_path, failure):
    sums = checkpoint / "SHA256SUMS"
    if failure == "missing_manifest":
        sums.unlink()
    elif failure.endswith("_hash"):
        omitted = "config.json" if failure == "missing_config_hash" else "model-00002.safetensors"
        sums.write_text(
            "".join(line for line in sums.read_text().splitlines(keepends=True) if not line.endswith(omitted + "\n"))
        )
    else:
        with (checkpoint / "model-00001.safetensors").open("ab") as stream:
            stream.write(b"corruption")
    destination = tmp_path / "bf16"
    with pytest.raises(ValueError, match="incomplete|checksum mismatch"):
        load_script("prepare_activevln_bf16").convert(checkpoint, destination)
    assert not destination.exists()


def test_conversion_refuses_to_overwrite_destination(checkpoint, tmp_path):
    destination = tmp_path / "bf16"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("existing output")
    with pytest.raises(FileExistsError):
        load_script("prepare_activevln_bf16").convert(checkpoint, destination)
    assert marker.read_text() == "existing output"


def test_failed_tensor_conversion_has_no_completion_marker(checkpoint, tmp_path, monkeypatch):
    import safetensors.numpy

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(safetensors.numpy, "save_file", fail)
    destination = tmp_path / "bf16"
    with pytest.raises(OSError, match="disk full"):
        load_script("prepare_activevln_bf16").convert(checkpoint, destination)
    assert not (destination / "SHA256SUMS").exists()


def test_dds_reexec_preserves_unified_owner_entrypoint(tmp_path):
    package = "embodirun_go2_owner"
    # A fresh process detects accidental imports of the Host from the vendor
    # entrypoint. Intercept execve so no native library or SDK is ever loaded.
    pythonpath = REPO / "integrations/go2_owner/src"
    library = tmp_path / "libddsc.so.0"
    library.write_bytes(b"test DDS library without native symbols")
    code = textwrap.dedent("""
        import importlib, os, sys
        from unittest.mock import patch
        package, directory = sys.argv[1:]
        loader = importlib.import_module(package + '.control.sdk_loader')
        importlib.import_module(package + '.cli')
        assert 'embodirun' not in sys.modules
        assert 'torch' not in sys.modules
        arguments = ['--config', '/tmp/owner.json']
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/old/dds'}):
            with patch.object(os, 'execve', side_effect=SystemExit(73)) as execute:
                try:
                    loader.ensure_cyclonedds_library_dir(directory, reexec_args=arguments)
                except SystemExit as error:
                    assert error.code == 73
                else:
                    raise AssertionError('DDS selection did not re-exec')
                executable, argv, environment = execute.call_args.args
                assert executable == sys.executable
                assert argv == [sys.executable, '-m', package, *arguments]
                assert environment['LD_LIBRARY_PATH'] == os.path.realpath(directory) + os.pathsep + '/old/dds'
    """)
    subprocess.run(
        [sys.executable, "-S", "-c", code, package, str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(pythonpath)),
        check=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("failure", [None, "range", "hash"])
def test_download_resume_checks_range_and_hash_before_publication(tmp_path, monkeypatch, failure):
    module = load_script("download_navigation_models")
    content = b"downloaded model bytes"
    offset = 7
    partial = tmp_path / "model.safetensors.partial"
    partial.write_bytes(content[:offset])
    item = {
        "rfilename": "model.safetensors",
        "size": len(content),
        "lfs": {"sha256": "0" * 64 if failure == "hash" else hashlib.sha256(content).hexdigest()},
    }

    class Response(io.BytesIO):
        def __init__(self, body, status=200, headers=None):
            super().__init__(body)
            self.status = status
            self.headers = headers or {}

    def open_url(request, **kwargs):
        assert request.unredirected_hdrs["Authorization"] == "Bearer test-read-token"
        assert "Authorization" not in request.headers
        if "/api/" in request.full_url:
            return Response(json.dumps({"sha": "a" * 40, "siblings": [item]}).encode())
        assert request.headers["Range"] == f"bytes={offset}-"
        start = 0 if failure == "range" else offset
        return Response(content[offset:], 206, {"Content-Range": f"bytes {start}-{len(content) - 1}/{len(content)}"})

    monkeypatch.setattr(module.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: "test-read-token"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["download", "--repo", "test/model", "--output", str(tmp_path), "--workers", "1", "--authenticated"],
    )
    if failure:
        with pytest.raises(RuntimeError, match="range|hash mismatch"):
            module.main()
        assert not (tmp_path / "SHA256SUMS").exists()
        assert not (tmp_path / "model.safetensors").exists()
        assert partial.read_bytes() == (content[:offset] if failure == "range" else content)
    else:
        module.main()
        assert (tmp_path / "model.safetensors").read_bytes() == content
        assert not partial.exists()
        assert (tmp_path / "SHA256SUMS").read_text() == f"{hashlib.sha256(content).hexdigest()}  model.safetensors\n"
