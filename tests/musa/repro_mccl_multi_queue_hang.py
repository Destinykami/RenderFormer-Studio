"""Standalone 8-GPU RenderFormer DDP + MCCL multi-stream workload.

Supports two workloads:
1. ``renderformer`` (default): instantiates the real 205M ``RenderFormerModel``
   from ``configs/model/v1/200m_full_attention.yaml`` with Triton RMSNorm,
   SwiGLU, and RoPE kernels, dual gradient checkpointing (view-independent and
   view-dependent transformers), PyTorch SDPA, and L1+LPIPS loss, driven purely
   by in-memory synthetic batches (no H5 files or pretrained weights required).
2. ``synthetic``: lightweight 12-layer Transformer with Triton ops and explicit
   ``zero_()`` memset for fast sanity checks.

Under ``MUSA_FORCE_SINGLE_QUEUE=1``, all ranks complete deterministically with
exact parameter parity across ranks. Without ``MUSA_FORCE_SINGLE_QUEUE=1``,
multi-queue context switching between ``Stream#0`` (compute) and ``Stream#3``
(``mcclKernel_AllReduce_RING_SIMPLE_Sum_float``) intermittently deadlocks with
``XID=1000007`` (``ExceptionType: ContextSwitchTimeout``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import yaml

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _select_device_and_backend() -> tuple[torch.device, str, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        import torch_musa  # noqa: F401

        if torch.musa.is_available():
            torch.musa.set_device(local_rank)
            return torch.device(f"musa:{local_rank}"), "mccl", local_rank
    except ImportError:
        pass

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), "nccl", local_rank

    return torch.device("cpu"), "gloo", local_rank


def _synchronize(device: torch.device) -> None:
    if device.type == "musa":
        torch.musa.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_renderformer_model(device: torch.device) -> nn.Module:
    os.environ.setdefault("USE_SDPA", "1")
    from renderformer.modeling import RenderFormerModel
    from renderformer.models.config import RenderTransformerConfig
    from renderformer.ops.apply_kernels import apply_kernels

    cfg_path = REPO_ROOT / "configs/model/v1/200m_full_attention.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    model_cfg = RenderTransformerConfig(**raw["model_config"])
    model = RenderFormerModel(model_cfg)
    if device.type in ("musa", "cuda"):
        model = apply_kernels(
            model, use_rms_norm=True, use_swiglu=True, use_rope=True, verbose=False
        )
    model.transformer.grad_checkpointing = True
    model.view_transformer.transformer.grad_checkpointing = True
    return model.to(device)


def _run_renderformer_workload(
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    rank: int,
    world_size: int,
) -> tuple[float, float, float]:
    from renderformer.models.ray_generator import RayGenerator

    torch.manual_seed(3407)
    model = _build_renderformer_model(device)
    use_gpu = device.type in ("musa", "cuda")
    if use_gpu:
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=5e-5, weight_decay=0.01)
    ray_generator = RayGenerator().to(device)

    lpips_criterion = None
    if not args.no_lpips:
        try:
            import lpips

            lpips_criterion = lpips.LPIPS(net="vgg", verbose=False).to(device)
        except Exception as exc:
            if rank == 0:
                print(f"[R0] LPIPS unavailable ({exc}); running L1-only loss", flush=True)

    dtype = torch.bfloat16 if use_gpu else torch.float32
    bs = args.batch_size
    nv = args.num_views
    pad_len = args.padding_length
    res = args.img_resolution

    # Pre-allocate synthetic buffers. When `h2d_copy` is enabled, keep the ~890 MB
    # texture tensor on host CPU and stream it H2D every step (matching real RF1
    # DataLoader PCIe traffic), paired with per-rank random jitter so ranks enter
    # backward and Stream#3 MCCL AllReduce out of phase.
    torch.manual_seed(3407 + rank)
    tris = torch.randn(bs, pad_len, 9, device=device, dtype=dtype) * 0.25
    textures_cpu = (
        torch.rand(bs, pad_len, 13, 32, 32, dtype=dtype)
        if args.h2d_copy
        else None
    )
    textures_dev = (
        None
        if args.h2d_copy
        else torch.rand(bs, pad_len, 13, 32, 32, device=device, dtype=dtype)
    )
    vns = F.normalize(torch.randn(bs, pad_len, 3, 3, device=device, dtype=dtype), dim=-1).reshape(bs, pad_len, 9)
    mvp_zero = torch.zeros(bs, 16, device=device, dtype=dtype)
    tri_vpos_view_tf = tris.float().unsqueeze(1).expand(bs, nv, pad_len, 9).contiguous()

    c2w = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).expand(bs, nv, 4, 4).clone()
    c2w[..., 2, 3] = 2.0
    fov = torch.full((bs, nv, 1), 45.0 / 180.0 * torch.pi, device=device, dtype=torch.float32)
    with torch.no_grad():
        rays_o, rays_d = ray_generator(c2w, fov, res)
    gt_imgs = torch.rand(bs * nv, 3, res, res, device=device, dtype=dtype)
    tri_idx = torch.arange(pad_len, device=device).unsqueeze(0)
    jitter_rng = random.Random(202601 + rank * 131)

    _synchronize(device)
    dist.barrier()

    t_start = time.perf_counter()
    last_norm = 0.0
    for step in range(args.steps):
        if args.jitter_max_ms > 0.0:
            time.sleep(jitter_rng.uniform(0.0, args.jitter_max_ms) / 1000.0)

        if textures_cpu is not None:
            textures = textures_cpu.to(device, non_blocking=True)
        else:
            assert textures_dev is not None
            textures = textures_dev

        # Varying active triangle counts per sample/rank (132..4170) mirrors stage-02 H5 data.
        gen = torch.Generator(device="cpu").manual_seed(10000 + step * 97 + rank)
        n_valid = torch.randint(132, min(4171, pad_len + 1), (bs, 1), generator=gen).to(device)
        valid_mask = tri_idx < n_valid

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_gpu):
            rendered_imgs = ddp_model(
                tris,
                textures,
                mvp_zero,
                valid_mask,
                vns,
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=tri_vpos_view_tf,
                tf32_view_tf=False,
            )

            pred = rendered_imgs.reshape(-1, *rendered_imgs.shape[2:])[:, :3]
            loss_l1 = F.l1_loss(pred, gt_imgs)
            if lpips_criterion is not None:
                pred_clamp = (pred / 0.30103).clamp(0.0, 1.0)
                gt_clamp = gt_imgs.clamp(0.0, 1.0)
                loss_lpips = lpips_criterion(pred_clamp, gt_clamp, normalize=True).mean()
                loss = loss_l1 + 0.05 * loss_lpips
            else:
                loss = loss_l1

        loss.backward()

        # Mirrors `train_rf1.py:345-349`: grad clip + host `.isnan()` sync point.
        total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        if bool(total_norm.isnan().item()):
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_norm = float(total_norm.item())

        if rank == 0 and (
            (step + 1) % args.heartbeat_interval == 0 or step + 1 == args.steps
        ):
            elapsed = time.perf_counter() - t_start
            ms_per_step = (elapsed / (step + 1)) * 1000.0
            print(
                f"[R0] step={step + 1}/{args.steps} "
                f"grad_norm={last_norm:.6f} ms_per_step={ms_per_step:.1f} "
                f"single_queue={os.environ.get('MUSA_FORCE_SINGLE_QUEUE', '<unset>')}",
                flush=True,
            )

    _synchronize(device)
    elapsed = time.perf_counter() - t_start

    checksum = torch.zeros(1, device=device, dtype=torch.float64)
    for p in model.parameters():
        checksum += p.detach().double().sum()
    checksum_min = checksum.clone()
    checksum_max = checksum.clone()
    dist.all_reduce(checksum_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(checksum_max, op=dist.ReduceOp.MAX)
    max_param_spread = float((checksum_max - checksum_min).abs().item())
    return elapsed, last_norm, max_param_spread


class _SyntheticBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def _impl(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scratch = torch.empty_like(h)
        scratch.zero_()
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h)) + scratch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.requires_grad:
            return checkpoint(self._impl, x, use_reentrant=False)
        return self._impl(x)


def _run_synthetic_workload(
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    rank: int,
    world_size: int,
) -> tuple[float, float, float]:
    torch.manual_seed(2026)
    use_gpu = device.type in ("musa", "cuda")
    model = nn.Sequential(*[_SyntheticBlock(768, 3072) for _ in range(12)]).to(device)
    ddp_model = DDP(model, device_ids=[local_rank]) if use_gpu else DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=5e-5, weight_decay=0.01)
    dtype = torch.bfloat16 if use_gpu else torch.float32

    _synchronize(device)
    dist.barrier()

    t_start = time.perf_counter()
    last_norm = 0.0
    for step in range(args.steps):
        gen = torch.Generator(device="cpu").manual_seed(10000 + step * 97 + rank)
        x = torch.randn(2, 2048, 768, generator=gen).to(device)
        target = torch.randn(2, 2048, 768, generator=gen).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_gpu):
            loss = F.l1_loss(ddp_model(x.requires_grad_(True)), target)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        if not bool(total_norm.isnan().item()):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_norm = float(total_norm.item())

        if rank == 0 and (
            (step + 1) % args.heartbeat_interval == 0 or step + 1 == args.steps
        ):
            elapsed = time.perf_counter() - t_start
            print(
                f"[R0] step={step + 1}/{args.steps} "
                f"grad_norm={last_norm:.6f} ms_per_step={(elapsed / (step + 1)) * 1000.0:.1f} "
                f"single_queue={os.environ.get('MUSA_FORCE_SINGLE_QUEUE', '<unset>')}",
                flush=True,
            )

    _synchronize(device)
    elapsed = time.perf_counter() - t_start

    checksum = torch.zeros(1, device=device, dtype=torch.float64)
    for p in model.parameters():
        checksum += p.detach().double().sum()
    checksum_min = checksum.clone()
    checksum_max = checksum.clone()
    dist.all_reduce(checksum_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(checksum_max, op=dist.ReduceOp.MAX)
    return elapsed, last_norm, float((checksum_max - checksum_min).abs().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("renderformer", "synthetic"),
        default="renderformer",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--padding-length", type=int, default=4192)
    parser.add_argument("--img-resolution", type=int, default=512)
    parser.add_argument("--no-lpips", action="store_true", default=False)
    parser.add_argument(
        "--h2d-copy",
        action="store_true",
        default=False,
        help="Stream ~890 MB texture tensor CPU->GPU asynchronously each step.",
    )
    parser.add_argument(
        "--jitter-max-ms",
        type=float,
        default=0.0,
        help="Random per-rank pre-step sleep jitter in milliseconds (e.g. 80.0).",
    )
    parser.add_argument("--heartbeat-interval", type=int, default=5)
    parser.add_argument("--summary-json", type=str, default="")
    args = parser.parse_args()

    device, backend, local_rank = _select_device_and_backend()
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        if args.workload == "renderformer":
            elapsed, last_norm, max_param_spread = _run_renderformer_workload(
                args, device, local_rank, rank, world_size
            )
        else:
            elapsed, last_norm, max_param_spread = _run_synthetic_workload(
                args, device, local_rank, rank, world_size
            )

        if rank == 0 and args.summary_json:
            payload = {
                "workload": args.workload,
                "steps": args.steps,
                "world_size": world_size,
                "backend": backend,
                "device_type": device.type,
                "musa_force_single_queue": os.environ.get("MUSA_FORCE_SINGLE_QUEUE"),
                "elapsed_sec": round(elapsed, 4),
                "ms_per_step": round((elapsed / max(args.steps, 1)) * 1000.0, 2),
                "last_grad_norm": round(last_norm, 6),
                "max_param_checksum_spread": max_param_spread,
            }
            with open(args.summary_json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
