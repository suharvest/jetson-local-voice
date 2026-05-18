"""ovs-agent CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from pathlib import Path

from .config import Config, load_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_config_path(app_name: str) -> Path:
    # agent/openvoicestream_agent/cli.py -> agent/
    agent_root = Path(__file__).resolve().parents[1]
    return agent_root / "apps" / app_name / "config.yaml"


def _load_app_class(app_name: str):  # noqa: ANN001
    """Dynamic import. Looks for `agent.apps.<name>.app:App`.

    Falls back to `apps.<name>.app:App` (when running from agent/ root).
    """
    candidates = [f"agent.apps.{app_name}.app", f"apps.{app_name}.app"]
    last_err: Exception | None = None
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            last_err = e
            continue
        cls = (
            getattr(mod, "App", None)
            or getattr(mod, "MultiModeApp", None)
            or getattr(mod, "DialogueApp", None)
        )
        if cls is None:
            raise ImportError(f"{mod_name} found but no `App` symbol")
        return cls
    raise ImportError(
        f"could not import app '{app_name}': last error: {last_err}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ovs-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an app")
    p_run.add_argument(
        "app_name",
        nargs="?",
        default="multi_mode",
        help="app under agent/apps/, e.g. 'multi_mode' (default)",
    )
    p_run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to YAML config (default: agent/apps/<app>/config.yaml)",
    )

    args = parser.parse_args(argv)
    if args.cmd != "run":  # pragma: no cover - argparse handles this
        parser.print_help()
        return 2

    cfg_path = args.config or _default_config_path(args.app_name)
    if not cfg_path.exists():
        # Allow running without a yaml -- fall back to defaults.
        if args.config is not None:
            print(f"config not found: {cfg_path}", file=sys.stderr)
            return 1
        cfg = Config()
    else:
        cfg = load_config(cfg_path)
    _setup_logging(cfg.log_level)

    app_cls = _load_app_class(args.app_name)
    app = app_cls(cfg)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
