"""Shared paths and opt-in guards for long-running stages.

Python port of ``code/_shared/paths.R`` and ``code/_shared/analysis_paths.R``.

The package root is ``Risk/hadi_model`` (the parent of ``code_py``). As in the R
code, the data and output roots can be overridden with the ``LDYG_DATA_ROOT`` and
``LDYG_OUTPUT_ROOT`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("LDYG_DATA_ROOT", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("LDYG_OUTPUT_ROOT", REPO_ROOT / "output"))
ANALYSIS_OUTPUT_ROOT = Path(
    os.environ.get("LDYG_ANALYSIS_OUTPUT_ROOT", OUTPUT_ROOT / "analysis")
)


def path_processed(*parts: str) -> Path:
    return DATA_ROOT.joinpath("processed", *parts)


def path_intermediate(*parts: str) -> Path:
    return DATA_ROOT.joinpath("intermediate", *parts)


def path_raw(*parts: str) -> Path:
    return DATA_ROOT.joinpath("raw", *parts)


def path_analysis_output(*parts: str) -> Path:
    return ANALYSIS_OUTPUT_ROOT.joinpath(*parts)


def assert_files_exist(paths: Iterable[os.PathLike] | os.PathLike) -> None:
    """Raise if any of ``paths`` is missing (R: ``assert_files_exist``)."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))


def allow_heavy(flag: str) -> bool:
    """True when environment variable ``flag`` is set to "true" (case-insensitive)."""
    return os.environ.get(flag, "false").strip().lower() == "true"


def stop_unless_heavy_allowed(flag: str, description: str) -> None:
    if not allow_heavy(flag):
        raise RuntimeError(
            f"{description}\nSet {flag}=true only when you intentionally want to "
            "run this long-running stage."
        )
