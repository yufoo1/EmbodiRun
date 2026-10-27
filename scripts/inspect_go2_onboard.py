#!/usr/bin/env python3
"""Read-only Orin environment and RealSense inventory; supports Python 3.8.

Run --camera with the vendor SDK Python, and --torch with the inference Python.
No camera stream, robot connection, installation, or motion is started.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import sys
from pathlib import Path


def camera_inventory():
    import pyrealsense2 as rs

    cameras = []
    agent_root = Path(__file__).resolve().parents[1] / "src" / "embodirun" / "robots" / "unitree"
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        scale = float(device.first_depth_sensor().get_depth_scale())
        profiles = {"color": set(), "depth": set()}
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                    kind = "color"
                elif profile.stream_type() == rs.stream.depth and profile.format() == rs.format.z16:
                    kind = "depth"
                else:
                    continue
                video = profile.as_video_stream_profile()
                profiles[kind].add((video.width(), video.height(), profile.fps()))
        supported = (640, 480, 15) in profiles["color"] & profiles["depth"]
        command = [
            "env",
            f"PYTHONPATH={agent_root}",
            sys.executable,
            "-m",
            "go2.agent.camera",
            "--backend",
            "realsense",
            "--profile",
            "640x480@15",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--serial",
            serial,
            "--depth-calibrated",
            "--depth-scale",
            str(scale),
        ]
        cameras.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": serial,
                "native_depth_scale_m": scale,
                "rgbd_profiles": [f"{w}x{h}@{fps}" for w, h, fps in sorted(profiles["color"] & profiles["depth"])],
                "launch_command": shlex.join(command) if supported else None,
            }
        )
    if not cameras:
        raise RuntimeError("no RealSense device found")
    return cameras


def torch_inventory():
    import torch
    import torchvision

    available = torch.cuda.is_available()
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    supported_version = version is not None and tuple(map(int, version.groups())) >= (2, 5)
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": available,
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
        "minimum_versions_met": sys.version_info >= (3, 10) and supported_version,
        "note": "Inventory only; checkpoint loading, memory and inference still require validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="enumerate RealSense profiles without streaming")
    parser.add_argument("--torch", action="store_true", help="inspect the selected Python's CUDA inference stack")
    args = parser.parse_args()
    l4t = Path("/etc/nv_tegra_release")
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "l4t": l4t.read_text().strip() if l4t.is_file() else None,
    }
    failed = False
    for enabled, name, inspect in ((args.camera, "camera", camera_inventory), (args.torch, "torch", torch_inventory)):
        if enabled:
            try:
                report[name] = inspect()
                if name == "camera":
                    failed |= not any(item["launch_command"] for item in report[name])
                else:
                    failed |= not (report[name]["cuda_available"] and report[name]["minimum_versions_met"])
            except Exception as error:
                report[name] = {"error": f"{type(error).__name__}: {error}"}
                failed = True
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
