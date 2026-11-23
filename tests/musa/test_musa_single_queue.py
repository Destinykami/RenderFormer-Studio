"""Unit and integration tests for `MUSA_FORCE_SINGLE_QUEUE=1` multi-GPU training.

Covers four angles for delivery verification:
1. Launcher guard (`scripts/train/_common.sh --print`): confirms multi-GPU MUSA
   launches automatically export `MUSA_FORCE_SINGLE_QUEUE=1` unless explicitly
   overridden by the caller, while single-GPU or non-MUSA runs stay untouched.
2. `.mudmp` forensic signature parser: verifies the dual-queue deadlock signature
   (`Stream#0` on `hw=0x...3070` vs `Stream#3` on `hw=0x...3370` waiting on
   `mcclKernel_AllReduce_RING_SIMPLE_Sum_float` + 2.0s staggered timeout) so
   post-mortem `.mudmp` evidence is machine-checked and self-documenting.
3. Positive multi-GPU DDP + MCCL verification (`MUSA_FORCE_SINGLE_QUEUE=1`):
   runs the dual-stream DDP + checkpointing + Triton + `zero_()` workload across
   all visible MUSA GPUs and asserts zero hang/timeout and exact cross-rank
   weight parity.
4. Optional negative stress probe (`--run-negative-repro`, skipped by default):
   runs the identical workload with `MUSA_FORCE_SINGLE_QUEUE` unset under a
   host-side heartbeat watchdog so engineers can probe the stochastic
   multi-queue `csw_timeout` deadlock without RenderFormer datasets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_SCRIPT = (
    REPO_ROOT / "scripts/train/recipes/v1/base_full_attention_stage_02_4k_512res.sh"
)
WORKER_SCRIPT = Path(__file__).resolve().parent / "repro_mccl_multi_queue_hang.py"


def _musa_device_count() -> int:
    try:
        import torch
        import torch_musa  # noqa: F401

        if torch.musa.is_available():
            return int(torch.musa.device_count())
    except ImportError:
        pass
    return 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _kill_process_tree(leader_pid: int, marker: str) -> None:
    """Terminate both the torchrun process group and any detached worker ranks."""
    try:
        os.killpg(leader_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # `torch.distributed.run` may place child workers into their own process
    # groups, so also sweep `/proc` for any process carrying our unique marker.
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().decode("utf-8", errors="ignore")
            except OSError:
                continue
            if marker in cmdline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _run_torchrun_with_watchdog(
    *,
    nproc: int,
    steps: int,
    env_overrides: dict[str, str | None],
    summary_path: Path,
    stall_timeout_sec: float = 45.0,
    startup_timeout_sec: float = 90.0,
    extra_worker_args: list[str] | None = None,
) -> tuple[int, bool, list[str]]:
    """Launch `torchrun` in its own process group and monitor stdout heartbeats."""
    env = os.environ.copy()
    for key, val in env_overrides.items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--master-port",
        str(_free_port()),
        "--nproc-per-node",
        str(nproc),
        str(WORKER_SCRIPT),
        "--workload",
        "renderformer",
        "--steps",
        str(steps),
        "--heartbeat-interval",
        "5",
        "--summary-json",
        str(summary_path),
    ]
    if extra_worker_args:
        cmd.extend(extra_worker_args)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None

    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for raw_line in iter(proc.stdout.readline, ""):
            line_queue.put(raw_line.rstrip("\r\n"))
        line_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    lines: list[str] = []
    saw_first_step = False
    last_activity = time.monotonic()
    eof_seen = False

    try:
        while True:
            try:
                item = line_queue.get(timeout=0.25)
                if item is None:
                    eof_seen = True
                else:
                    last_activity = time.monotonic()
                    lines.append(item)
                    print(item, flush=True)
                    if "[R0] step=" in item:
                        saw_first_step = True
            except queue.Empty:
                pass

            rc = proc.poll()
            if rc is not None and eof_seen:
                return rc, False, lines

            limit = stall_timeout_sec if saw_first_step else startup_timeout_sec
            if time.monotonic() - last_activity > limit:
                _kill_process_tree(proc.pid, str(summary_path))
                proc.wait(timeout=15)
                return -9, True, lines
    finally:
        if proc.poll() is None:
            _kill_process_tree(proc.pid, str(summary_path))
            proc.wait(timeout=10)
        else:
            _kill_process_tree(proc.pid, str(summary_path))


class TestLauncherSingleQueueGuard:
    """Fast CPU-only unit tests for `scripts/train/_common.sh`."""

    def _print_recipe(self, env_updates: dict[str, str | None], tmp_path: Path) -> str:
        env = os.environ.copy()
        env.pop("MUSA_FORCE_SINGLE_QUEUE", None)
        env["DATASET_LIST"] = str(tmp_path / "paths.txt")
        env["OUTPUT_ROOT"] = str(tmp_path / "runs")
        env["INIT_WEIGHTS"] = str(tmp_path / "init")
        for key, val in env_updates.items():
            if val is None:
                env.pop(key, None)
            else:
                env[key] = val

        res = subprocess.run(
            ["bash", str(RECIPE_SCRIPT), "--print"],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout

    def test_multi_gpu_musa_defaults_single_queue_on(self, tmp_path: Path) -> None:
        out = self._print_recipe(
            {"MUSA_RUNTIME_DETECT": "1", "NPROC_PER_NODE": "8"}, tmp_path
        )
        assert "MUSA_FORCE_SINGLE_QUEUE=1 " in out

    def test_multi_node_single_local_gpu_musa_defaults_single_queue_on(
        self, tmp_path: Path
    ) -> None:
        out = self._print_recipe(
            {
                "MUSA_RUNTIME_DETECT": "1",
                "NPROC_PER_NODE": "1",
                "NNODES": "2",
                "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
            },
            tmp_path,
        )
        assert "MUSA_FORCE_SINGLE_QUEUE=1 " in out

    def test_explicit_caller_override_is_preserved(self, tmp_path: Path) -> None:
        out = self._print_recipe(
            {
                "MUSA_RUNTIME_DETECT": "1",
                "NPROC_PER_NODE": "8",
                "MUSA_FORCE_SINGLE_QUEUE": "0",
            },
            tmp_path,
        )
        assert "MUSA_FORCE_SINGLE_QUEUE=0 " in out

    def test_single_gpu_or_non_musa_does_not_inject_flag(self, tmp_path: Path) -> None:
        out_1gpu = self._print_recipe(
            {"MUSA_RUNTIME_DETECT": "1", "NPROC_PER_NODE": "1"}, tmp_path
        )
        out_cuda = self._print_recipe(
            {"MUSA_RUNTIME_DETECT": "0", "NPROC_PER_NODE": "8"}, tmp_path
        )
        assert "MUSA_FORCE_SINGLE_QUEUE=" not in out_1gpu
        assert "MUSA_FORCE_SINGLE_QUEUE=" not in out_cuda


def parse_mudmp_dual_queue_signature(mudmp_texts: list[str]) -> dict[str, object]:
    """Extract the hardware-queue and stream-wait topology from MUSA `.mudmp` texts.

    When `MUSA_FORCE_SINGLE_QUEUE` is NOT set, `Stream#0` (default compute) and
    `Stream#3` (MCCL collectives) bind to distinct hardware queues (e.g.
    `hw=0x81e1c03070` vs `hw=0x81e1c03370`). During the stochastic deadlock:
    - Slow ranks fault first on `Stream#0` (`ContextSwitchTimeout` on compute /
      `Memset` / `Record`) while `Stream#3` has queued `Barrier ... <-Stream#0`
      entries waiting on uncompleted `Stream#0` events;
    - Fast ranks fault ~2.0 seconds later on `Stream#3` inside
      `mcclKernel_AllReduce_RING_SIMPLE_Sum_float` spinning for the slow peers.
    """
    stream0_fault_devs: list[int] = []
    stream3_mccl_fault_devs: list[int] = []
    distinct_hw_queue_devs: list[int] = []
    cross_stream_barrier_devs: list[int] = []

    for txt in mudmp_texts:
        hdr = txt.splitlines()[0] if txt else ""
        m_hdr = re.search(
            r"MUSA_ERROR_LAUNCH_TIMEOUT in stream (\d+) of context \d+ of device (\d+)",
            hdr,
        )
        if not m_hdr:
            continue
        err_stream = int(m_hdr.group(1))
        dev = int(m_hdr.group(2))

        hw_s0 = re.search(r"Stream#0\s*\([^)]*hw=(0x[0-9a-fA-F]+)\[hw\]", txt)
        hw_s3 = re.search(r"Stream#3\s*\([^)]*hw=(0x[0-9a-fA-F]+)\[hw\]", txt)
        if hw_s0 and hw_s3 and hw_s0.group(1) != hw_s3.group(1):
            distinct_hw_queue_devs.append(dev)

        if re.search(r"waits:.*event=0x[0-9a-fA-F]+,val=\d+,cur=\d+\s*✗\s*<-Stream#0", txt):
            cross_stream_barrier_devs.append(dev)

        if err_stream == 0:
            stream0_fault_devs.append(dev)
        elif err_stream == 3 and "mcclKernel_AllReduce" in hdr:
            stream3_mccl_fault_devs.append(dev)

    return {
        "stream0_fault_devs": sorted(stream0_fault_devs),
        "stream3_mccl_fault_devs": sorted(stream3_mccl_fault_devs),
        "distinct_hw_queue_devs": sorted(distinct_hw_queue_devs),
        "cross_stream_barrier_devs": sorted(cross_stream_barrier_devs),
        "is_dual_queue_mccl_deadlock": bool(
            stream0_fault_devs
            and stream3_mccl_fault_devs
            and distinct_hw_queue_devs
            and cross_stream_barrier_devs
        ),
    }


def test_mudmp_forensic_signature_documents_dual_queue_deadlock() -> None:
    """CPU unit test verifying the exact `.mudmp` signature from 8-GPU crashes."""
    # Verbatim sanitized excerpts from worker35029 / worker35030 / worker35031 crashes.
    slow_rank_mudmp = """\
[2026-11-20_10:23:02.781] pid:1126414, Application: /usr/bin/python encountered MUSA_ERROR_LAUNCH_TIMEOUT in stream 0 of context 0 of device 1 when executing a command of type Command Record
ExceptionType: ContextSwitchTimeout
Device#1
└── Context#0
    ├── Stream#0 (tl=0x5618154217d0[tl], hw=0x81e1c03070[hw], total=48328) (default)
    │   ├── [Record      @0x7f4a237dabe0]  error       sig->(sem=0x81e1c03070[hw],val=48324)  waits: sem=0x81e1c03070[hw],val=48323,cur=48323 ✓
    │   ├── [Memset      @0x7f4a236b8f70]  error       sig->(sem=0x81e1c03070[hw],val=48324)  waits: (none)
    └── Stream#3 (tl=0x561821295d20[tl], hw=0x81e1c03370[hw], total=1166)
        ├── [Barrier     @0x7fb7d09e75d0]  submitted   sig->(sem=0x81e1c03370[hw],val=1165)  waits: event=0x1000cc00010,val=1543,cur=1540 ✗ <-Stream#0
        ├── [Dispatch    @0x7fb7d0973c40]  submitted   sig->(sem=0x81e1c03370[hw],val=1165)  waits: (none)  kernel=mcclKernel_AllReduce_RING_SIMPLE_Sum_float(mcclWorkElem)
"""
    fast_rank_mudmp = """\
[2026-11-20_10:23:04.779] pid:1126416, Application: /usr/bin/python encountered MUSA_ERROR_LAUNCH_TIMEOUT in stream 3 of context 0 of device 3 when executing a command of type Command Dispatch at mcclKernel_AllReduce_RING_SIMPLE_Sum_float<<<{28, 1, 1}, {512, 1, 1}>>>
ExceptionType: ContextSwitchTimeout
Device#3
└── Context#0
    ├── Stream#0 (tl=0x5623db8f94a0[tl], hw=0x81e1c03070[hw], total=48241) (default)
    └── Stream#3 (tl=0x5623e6979020[tl], hw=0x81e1c03370[hw], total=1162)
        ├── [Dispatch    @0x7fc948a1c2d0]  error       sig->(sem=0x81e1c03370[hw],val=1160)  waits: sem=0x81e1c03370[hw],val=1159,cur=1159 ✓  kernel=mcclKernel_AllReduce_RING_SIMPLE_Sum_float(mcclWorkElem)
        ├── [Barrier     @0x7fc9489d4d20]  error       sig->(sem=0x81e1c03370[hw],val=1160)  waits: event=0x1000cc00010,val=1541,cur=1540 ✗ <-Stream#0
"""
    diag = parse_mudmp_dual_queue_signature([slow_rank_mudmp, fast_rank_mudmp])
    assert diag["is_dual_queue_mccl_deadlock"] is True
    assert diag["stream0_fault_devs"] == [1]
    assert diag["stream3_mccl_fault_devs"] == [3]
    assert diag["distinct_hw_queue_devs"] == [1, 3]
    assert diag["cross_stream_barrier_devs"] == [1, 3]


def test_musa_ddp_overlap_with_single_queue(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Positive multi-GPU MUSA verification with `MUSA_FORCE_SINGLE_QUEUE=1`."""
    visible = _musa_device_count()
    if visible < 2:
        pytest.skip(f"Requires >=2 MUSA GPUs, found {visible}")

    requested_nproc = int(request.config.getoption("--musa-nproc"))
    nproc = requested_nproc if requested_nproc > 0 else visible
    steps = int(request.config.getoption("--musa-steps"))
    summary_path = tmp_path / "single_queue_summary.json"

    rc, stalled, _ = _run_torchrun_with_watchdog(
        nproc=nproc,
        steps=steps,
        env_overrides={"MUSA_FORCE_SINGLE_QUEUE": "1", "USE_SDPA": "1"},
        summary_path=summary_path,
        stall_timeout_sec=60.0,
        startup_timeout_sec=120.0,
        extra_worker_args=["--h2d-copy", "--jitter-max-ms", "80.0"],
    )

    assert not stalled, (
        f"DDP workload stalled with MUSA_FORCE_SINGLE_QUEUE=1 across {nproc} GPUs"
    )
    assert rc == 0, f"torchrun exited with code {rc}"
    assert summary_path.is_file(), "Missing rank-0 summary JSON"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["musa_force_single_queue"] == "1"
    assert summary["world_size"] == nproc
    assert summary["steps"] == steps
    assert summary["max_param_checksum_spread"] == 0.0


def test_reproduce_multi_queue_hang_without_single_queue(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Negative diagnostic reproduction with `MUSA_FORCE_SINGLE_QUEUE` unset."""
    if not request.config.getoption("--run-negative-repro"):
        pytest.skip("Pass --run-negative-repro to run the multi-queue hang diagnostic")

    visible = _musa_device_count()
    if visible < 2:
        pytest.skip(f"Requires >=2 MUSA GPUs, found {visible}")

    requested_nproc = int(request.config.getoption("--musa-nproc"))
    nproc = requested_nproc if requested_nproc > 0 else visible
    steps = int(request.config.getoption("--negative-steps"))
    summary_path = tmp_path / "multi_queue_summary.json"

    rc, stalled, lines = _run_torchrun_with_watchdog(
        nproc=nproc,
        steps=steps,
        env_overrides={"MUSA_FORCE_SINGLE_QUEUE": None, "USE_SDPA": "1"},
        summary_path=summary_path,
        stall_timeout_sec=60.0,
        startup_timeout_sec=120.0,
        extra_worker_args=["--h2d-copy", "--jitter-max-ms", "80.0"],
    )

    last_hb = next((l for l in reversed(lines) if "[R0] step=" in l), "<none>")
    if stalled:
        print(
            f"[REPRO CONFIRMED] Multi-queue stall detected without "
            f"MUSA_FORCE_SINGLE_QUEUE=1 (last heartbeat: {last_hb})",
            flush=True,
        )

    # This diagnostic exists to demonstrate the unmitigated failure mode; if a
    # future MUSA driver/firmware release fixes multi-queue scheduling, xfail
    # cleanly so engineers see that the workaround is no longer required.
    if not stalled and rc == 0:
        pytest.xfail(
            f"Completed {steps} steps without MUSA_FORCE_SINGLE_QUEUE=1; "
            "multi-queue deadlock did not trigger in this window or is fixed in driver."
        )
    assert stalled or rc != 0
