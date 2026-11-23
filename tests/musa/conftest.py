"""Pytest configuration for MUSA multi-GPU single-queue tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-negative-repro",
        action="store_true",
        default=False,
        help=(
            "Run the negative multi-queue reproduction workload without "
            "MUSA_FORCE_SINGLE_QUEUE=1 (may trigger MUSA csw_timeout)."
        ),
    )
    parser.addoption(
        "--musa-nproc",
        type=int,
        default=0,
        help="Override GPU worker count for MUSA DDP tests (0 = all visible MUSA GPUs).",
    )
    parser.addoption(
        "--musa-steps",
        type=int,
        default=20,
        help="Iterations for the positive single-queue RenderFormer DDP test (default: 20).",
    )
    parser.addoption(
        "--negative-steps",
        type=int,
        default=600,
        help="Max iterations for the negative multi-queue RenderFormer reproduction test (default: 600).",
    )
