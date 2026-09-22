"""Batched versions of the existing acoustic KDE forward model.

Scientific parameters, soft bins, rounding, and the straight-through histogram
gradient match :mod:`lib.forward`. Batching only changes execution and storage.
"""

from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn.functional as F

from .forward import (
    FIXED_POINT_INV_SCALE,
    FIXED_POINT_SCALE,
    GRID_SIZE_DEFAULT,
    KDE_DELTA,
    KDE_N_BINS,
    KDE_R_MIN,
    SIGMA_DEFAULT,
    STRICT_ATOMICS,
    VOXEL_DEFAULT,
    VS_DEFAULT,
)


@lru_cache(maxsize=256)
def _kernel(sigma, delta_bin, device, dtype):
    # A cache may first be populated during inference and later used in autograd.
    # Ordinary no-grad tensors (not inference tensors) support both usages.
    with torch.inference_mode(False), torch.no_grad():
        half = int(math.ceil(5.0 * sigma / delta_bin))
        d = torch.arange(-half, half + 1, device=device, dtype=dtype) * delta_bin
        k = d * torch.exp(-d * d / (2.0 * sigma * sigma))
        return k.view(1, 1, -1), half


@lru_cache(maxsize=64)
def _time_samples(t_start_idx, n_time_sub, delta_t, vs, delta_bin, r_min,
                  n_bins, device, dtype):
    with torch.inference_mode(False), torch.no_grad():
        t_rel = torch.arange(n_time_sub, device=device, dtype=dtype)
        u = vs * (t_start_idx + t_rel) * delta_t
        pos_t = (u - r_min) / delta_bin
        j0 = pos_t.floor().long().clamp_(0, n_bins - 2)
        beta = pos_t - j0.to(dtype)
        # Only cache creation synchronizes. Bounds never depend on a sensor or
        # trainable tensor, and let each sigma use exactly the needed support.
        bounds = torch.stack((j0.min(), j0.max())).tolist() if n_time_sub else (0, 0)
        return j0, beta, bounds[0], bounds[1]


def clear_kde_batch_cache():
    """Release cached kernels and fixed time-index tensors, including GPU ones."""
    _kernel.cache_clear()
    _time_samples.cache_clear()


def kde_acoustic_sim_batch(sens_pos_batch, src_pos, Pc, t_start_idx,
                           n_time_sub, delta_t, vs=VS_DEFAULT,
                           sigma=SIGMA_DEFAULT, delta_bin=None,
                           r_min=None, n_bins=None, crop=True, projector="torch"):
    """Simulate ``(B, 3)`` sensors against shared ``(K, 3)`` sources.

    Returns ``(B, n_time_sub)``. Sensor coordinates, source coordinates and
    intensities retain autograd support; acoustic/time arguments are fixed
    Python scalars, as used by the localization pipeline. Memory is O(B*K +
    B*n_bins), so callers should chunk large sensor/candidate sets.

    STRICT_ATOMICS uses exactly the reference's int64 histogram and float
    straight-through gradient. As with the reference, its gradient histogram
    uses floating point atomics; batching does not promise bitwise equality of
    convolution or gradient reductions across different batch sizes.

    With ``crop=True``, convolution only covers the sampled time-index range
    and its full kernel support. The histogram and its binning are unchanged;
    ``crop=False`` provides the full-distance-grid execution for comparison.
    ``projector="triton"`` selects the fused CUDA float32 sensor-gradient-only
    histogram. The default ``"torch"`` retains source and intensity gradients.
    """
    if sens_pos_batch.ndim != 2 or sens_pos_batch.shape[1] != 3:
        raise ValueError("sens_pos_batch must have shape (B, 3)")
    delta_bin = KDE_DELTA if delta_bin is None else delta_bin
    r_min_v = KDE_R_MIN if r_min is None else r_min
    n_bins_v = KDE_N_BINS if n_bins is None else n_bins
    device, dtype = Pc.device, Pc.dtype

    if projector == "triton":
        from .forward_triton import kde_histogram_triton
        h = kde_histogram_triton(sens_pos_batch, src_pos, Pc, delta_bin=delta_bin,
                                 r_min=r_min_v, n_bins=n_bins_v,
                                 strict_atomics=STRICT_ATOMICS)
    elif projector == "torch":
        h = _histogram_torch(sens_pos_batch, src_pos, Pc, delta_bin, r_min_v, n_bins_v)
    else:
        raise ValueError("projector must be 'torch' or 'triton'")

    k, half = _kernel(sigma, delta_bin, device, dtype)
    j0, beta, j_min, j_max = _time_samples(t_start_idx, n_time_sub, delta_t, vs,
                                          delta_bin, r_min_v, n_bins_v, device, dtype)
    if crop:
        left = max(0, j_min - half)
        right = min(n_bins_v, j_max + half + 2)
        h = h[:, left:right]
        j0 = j0 - left
    p_grid = F.conv1d(h.unsqueeze(1), k, padding=half)[:, 0, :]
    return (1.0 - beta).unsqueeze(0) * p_grid[:, j0] + beta.unsqueeze(0) * p_grid[:, j0 + 1]


def _histogram_torch(sens_pos_batch, src_pos, Pc, delta_bin, r_min_v, n_bins_v):
    """Original batched PyTorch projection, including the fixed-point STE."""
    device, dtype = Pc.device, Pc.dtype
    batch_size = sens_pos_batch.shape[0]

    diff = sens_pos_batch.unsqueeze(1) - src_pos.unsqueeze(0)
    r = torch.norm(diff, dim=2) + 1e-12
    w = Pc.unsqueeze(0) / (2.0 * r)
    pos = (r - r_min_v) / delta_bin
    i0 = pos.floor().long().clamp_(0, n_bins_v - 1)
    alpha = pos - i0.to(dtype)
    val0 = (1.0 - alpha) * w
    val1 = alpha * w

    if STRICT_ATOMICS:
        # Rounding is intentionally outside the graph. The reference's STE
        # differentiates val0/val1, not this integer projection.
        with torch.no_grad():
            v0 = val0 * FIXED_POINT_SCALE
            v1 = val1 * FIXED_POINT_SCALE
            q0 = torch.where(v0 >= 0.0, torch.floor(v0 + 0.5),
                             -torch.floor(-v0 + 0.5)).to(torch.int64)
            q1 = torch.where(v1 >= 0.0, torch.floor(v1 + 0.5),
                             -torch.floor(-v1 + 0.5)).to(torch.int64)
            h_q = torch.zeros((batch_size, n_bins_v + 1),
                              device=device, dtype=torch.int64)
            h_q.scatter_add_(1, i0, q0)
            h_q.scatter_add_(1, i0 + 1, q1)
            h_det = h_q[:, :n_bins_v].to(dtype) * FIXED_POINT_INV_SCALE
            del h_q, q0, q1, v0, v1

        if torch.is_grad_enabled() and (val0.requires_grad or val1.requires_grad):
            h_grad = torch.zeros((batch_size, n_bins_v + 1), device=device, dtype=dtype)
            h_grad.scatter_add_(1, i0, val0)
            h_grad.scatter_add_(1, i0 + 1, val1)
            h_grad = h_grad[:, :n_bins_v]
            h = h_det.detach() + (h_grad - h_grad.detach())
        else:
            h = h_det
    else:
        h = torch.zeros((batch_size, n_bins_v + 1), device=device, dtype=dtype)
        h.scatter_add_(1, i0, val0)
        h.scatter_add_(1, i0 + 1, val1)
        h = h[:, :n_bins_v]

    return h


def load_phantom_topk_fast(ckpt_path, device, keep_ratio=0.002,
                          grid_size=GRID_SIZE_DEFAULT, voxel_size=VOXEL_DEFAULT):
    """Reference top-K selection without materializing three full-grid arrays.

    A tiny NumPy coordinate table preserves the reference's float64-to-float32
    coordinate conversion exactly. Checkpoint optimizer tensors stay on CPU and
    are released before source selection, because they are not used here.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    Pc_raw = ckpt["Pc_state"].to(device).contiguous()
    del ckpt
    n_keep = int(Pc_raw.numel() * keep_ratio)
    _, idx_top = torch.topk(torch.abs(Pc_raw), n_keep)
    coords = (np.arange(grid_size) - (grid_size - 1) / 2.0) * voxel_size
    coords = torch.tensor(coords.astype(np.float32), device=device)
    ix = idx_top // (grid_size * grid_size)
    iy = (idx_top // grid_size) % grid_size
    iz = idx_top % grid_size
    xyz_src = torch.stack((coords[ix], coords[iy], coords[iz]), dim=1).contiguous()
    return xyz_src, Pc_raw[idx_top].contiguous()
