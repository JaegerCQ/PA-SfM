"""Batch the original KDE localization without reducing search or iteration counts.

Coarse candidate signals depend only on the source field, so compute them once
per process. Fine optimization batches independent sensor/start pairs; summing
their losses preserves each parameter row's original Adam gradient scale.
"""
import argparse
from functools import partial
import json
import os
import time
from pathlib import Path

from lib.runtime_config import DEFAULT_SEED, seed_everything

import numpy as np
import torch

from lib.forward import VS_DEFAULT
from lib.forward_batch import kde_acoustic_sim_batch, load_phantom_topk_fast

ROOT = Path(__file__).resolve().parent
SEED = seed_everything(DEFAULT_SEED)
FORWARD_BATCH = kde_acoustic_sim_batch


def neg_corr_loss_batch(pred, target):
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    return 1.0 - (p * t).sum(dim=-1) / (
        torch.linalg.vector_norm(p, dim=-1) * torch.linalg.vector_norm(t, dim=-1) + 1e-9)


@torch.no_grad()
def prepare_coarse(xyz_src, Pc, t_start, n_time, delta_t, batch_size=32):
    # Generate on CPU exactly as the legacy implementation does.
    axis = torch.arange(-0.16, 0.16, 0.02)
    mesh = torch.meshgrid(axis, axis, axis, indexing="ij")
    candidates = torch.stack([v.flatten() for v in mesh], dim=1).to(Pc.device)
    templates = []
    for start in range(0, len(candidates), batch_size):
        templates.append(FORWARD_BATCH(
            candidates[start:start + batch_size], xyz_src, Pc,
            t_start, n_time, delta_t, vs=VS_DEFAULT, sigma=0.5e-3))
    return candidates, torch.cat(templates)


@torch.no_grad()
def select_starts(candidates, templates, targets, k):
    # Use elementwise reductions rather than TF32 GEMM to preserve correlation
    # ordering close to the legacy scalar calculation.
    indices = torch.stack([
        neg_corr_loss_batch(templates, target).cpu().topk(k, largest=False).indices
        for target in targets
    ]).to(candidates.device)
    return candidates[indices]


def fine_search_batch(xyz_src, Pc, target_sig, start_pos, total_epochs, lr,
                      sigma_start, sigma_target, t_start, n_time, delta_t):
    """Independent Adam runs, with rows ordered (sensor, coarse-start)."""
    param = torch.nn.Parameter(start_pos.detach().clone())
    opt = torch.optim.Adam([param], lr=lr, foreach=False)
    for epoch in range(total_epochs):
        opt.zero_grad(set_to_none=True)
        sigma = sigma_start * (sigma_target / sigma_start) ** min(1.0, epoch / (total_epochs * 0.8))
        pred = FORWARD_BATCH(param, xyz_src, Pc, t_start, n_time, delta_t,
                                     vs=VS_DEFAULT, sigma=sigma)
        losses = neg_corr_loss_batch(pred, target_sig)
        losses.sum().backward()
        opt.step()
    # Match legacy selection: loss from just before the last Adam update, and
    # the position immediately after that update.
    return param.detach(), losses.detach()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal", required=True)
    ap.add_argument("--sensor_gt", required=True,
                    help="reference array for logging only; not pose1 ground truth")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_ratio", type=float, default=0.002)
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--total_epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--sigma_start", type=float, default=1.5e-3)
    ap.add_argument("--sigma_target", type=float, default=0.1e-3)
    ap.add_argument("--delta_t", type=float, default=100e-9)
    ap.add_argument("--t_start", type=int, default=500)
    ap.add_argument("--t_end", type=int, default=1000)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--sensor_limit", type=int, default=0)
    ap.add_argument("--sensor_ids", default="")
    ap.add_argument("--use_kde", action="store_true", help="compatibility flag; this backend always uses KDE")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--coarse_batch_size", type=int, default=32)
    ap.add_argument("--sensor_batch_size", type=int, default=4)
    ap.add_argument("--projector", choices=("torch", "triton"),
                    default=os.environ.get("LOCALIZATION_PROJECTOR", "triton"))
    args = ap.parse_args()
    if min(args.coarse_batch_size, args.sensor_batch_size, args.total_epochs, args.top_k) < 1:
        ap.error("batch sizes, total_epochs and top_k must be positive")
    if not 0 < args.keep_ratio <= 1 or args.t_end <= args.t_start:
        ap.error("invalid keep_ratio or time range")
    return args


def main():
    global FORWARD_BATCH
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    FORWARD_BATCH = partial(kde_acoustic_sim_batch, projector=args.projector)
    print(f"[INIT] device={device} backend=batch projector={args.projector} sensor_batch={args.sensor_batch_size} "
          f"coarse_batch={args.coarse_batch_size}", flush=True)
    xyz_src, Pc = load_phantom_topk_fast(args.ckpt, device, keep_ratio=args.keep_ratio)
    signals = np.loadtxt(args.signal)
    if args.downsample > 1:
        signals = signals[:, ::args.downsample]
    sensor_gt = np.loadtxt(args.sensor_gt)
    total = len(signals)
    if sensor_gt.shape != (total, 3):
        raise ValueError("Reference array must contain one 3D position per sensor")
    if args.sensor_ids.strip():
        sensor_ids = [int(x) for x in args.sensor_ids.split(",")]
    elif args.sensor_limit > 0:
        sensor_ids = list(range(min(args.sensor_limit, total)))
    else:
        sensor_ids = list(range(total))
    if not sensor_ids or min(sensor_ids) < 0 or max(sensor_ids) >= total:
        raise ValueError("sensor_ids must be nonempty and within the signal array")
    targets = torch.as_tensor(signals[sensor_ids, args.t_start:args.t_end],
                              dtype=Pc.dtype, device=device)
    n_time = args.t_end - args.t_start
    print(f"[INFO] kept {len(Pc)} source points; {len(sensor_ids)} sensors selected", flush=True)
    stage = time.perf_counter()
    candidates, templates = prepare_coarse(xyz_src, Pc, args.t_start, n_time,
                                           args.delta_t, args.coarse_batch_size)
    starts = select_starts(candidates, templates, targets, args.top_k)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    coarse_seconds = time.perf_counter() - stage
    print(f"[COARSE] {len(candidates)} shared candidates; {coarse_seconds:.3f}s", flush=True)
    preds = np.full((total, 3), np.nan, dtype=np.float64)
    correlations = np.full(total, np.nan)
    stage = time.perf_counter()
    for offset in range(0, len(sensor_ids), args.sensor_batch_size):
        subset = sensor_ids[offset:offset + args.sensor_batch_size]
        initial = starts[offset:offset + len(subset)].reshape(-1, 3)
        target_batch = targets[offset:offset + len(subset)].repeat_interleave(args.top_k, dim=0)
        positions, losses = fine_search_batch(
            xyz_src, Pc, target_batch, initial, args.total_epochs, args.lr,
            args.sigma_start, args.sigma_target, args.t_start, n_time, args.delta_t)
        positions = positions.reshape(len(subset), args.top_k, 3)
        losses = losses.reshape(len(subset), args.top_k)
        # As in the legacy `loss < best_loss` loop, ignore failed starts when
        # other starts for that sensor remain usable.
        losses = torch.where(torch.isfinite(losses), losses, torch.inf)
        best = losses.argmin(dim=1)
        row = torch.arange(len(subset), device=device)
        best_pos = positions[row, best].cpu().numpy()
        best_corr = (1.0 - losses[row, best]).cpu().numpy()
        if not np.isfinite(best_pos).all() or not np.isfinite(best_corr).all():
            raise RuntimeError("Non-finite localization result")
        preds[subset] = best_pos
        correlations[subset] = best_corr
        elapsed = time.perf_counter() - stage
        completed = offset + len(subset)
        eta = elapsed / completed * (len(sensor_ids) - completed)
        print(f"[{completed}/{len(sensor_ids)}] sensors={subset} "
              f"mean_corr={best_corr.mean():.6f} ({elapsed:.1f}s elapsed, eta {eta:.1f}s)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, preds, fmt="%.6f", header="Predicted x, y, z (meters); NaN = sensor not solved")
    report = {"backend": "batch", "arguments": vars(args), "coarse_seconds": coarse_seconds,
              "fine_seconds": time.perf_counter() - stage,
              "total_seconds": time.perf_counter() - started,
              "sensor_ids": sensor_ids, "correlations": correlations[sensor_ids].tolist(),
              "note": "Correlations match the legacy pre-final-update logging convention; reference positions are not ground truth."}
    out.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[DONE] saved {out}", flush=True)


if __name__ == "__main__":
    main()
