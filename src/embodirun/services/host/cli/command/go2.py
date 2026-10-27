"""Route Go2 policy bindings through the installed EmbodiRun command."""

from __future__ import annotations

import argparse
from typing import Any


def register(commands: Any) -> None:
    """Register the robot/model path without importing hardware dependencies."""
    parser = commands.add_parser("go2", help="Go2 navigation and edge services")
    bindings = parser.add_subparsers(dest="binding", required=True)
    activevln = bindings.add_parser("activevln", add_help=False, help="ActiveVLN navigation")
    activevln.add_argument("binding_args", nargs=argparse.REMAINDER)


def run(args: argparse.Namespace) -> int:
    """Invoke the binding without requiring deployment YAML."""
    from embodirun.bindings.unitree.go2.activevln.cli import main

    return main(args.binding_args)
