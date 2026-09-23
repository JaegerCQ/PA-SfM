"""Fused KDE histogram with a deterministic sensor-position backward pass.

This is an optional CUDA float32 projector, not a different acoustic model.
Only sensor coordinates may require gradients. Convolution and time sampling
remain the responsibility of the caller.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .forward import (
    FIXED_POINT_INV_SCALE, FIXED_POINT_SCALE, KDE_DELTA, KDE_N_BINS,
    KDE_R_MIN, STRICT_ATOMICS,
)


@triton.jit
def _geometry(sensors, sources, intensities, ids, sensor_id, valid,
              delta_bin, r_min, n_bins):
    dx = tl.load(sensors + 3 * sensor_id) - tl.load(sources + 3 * ids, valid, 0.0)
    dy = tl.load(sensors + 3 * sensor_id + 1) - tl.load(sources + 3 * ids + 1, valid, 0.0)
    dz = tl.load(sensors + 3 * sensor_id + 2) - tl.load(sources + 3 * ids + 2, valid, 0.0)
    # The CUDA three-element torch.norm reduction sums lanes (x,z), then y.
    # Matching this order and scalar-division's reciprocal multiplication keeps
    # the reference's sub-bin coordinates (and fixed-point bins) unchanged.
    norm = libdevice.sqrt_rn((dx * dx + dz * dz) + dy * dy)
    r = norm + 1.0e-12
    pc = tl.load(intensities + ids, valid, 0.0)
    w = tl.div_rn(pc, 2.0 * r)
    pos = (r - r_min) * tl.div_rn(1.0, delta_bin)
    i0 = tl.minimum(tl.maximum(tl.floor(pos).to(tl.int32), 0), n_bins - 1)
    alpha = pos - i0.to(tl.float32)
    return dx, dy, dz, norm, r, w, i0, alpha


@triton.jit
def _project(sensors, sources, intensities, histogram, n_sources,
             n_bins, delta_bin, r_min, fixed_scale,
             STRICT: tl.constexpr, BLOCK: tl.constexpr,
             N_SENSORS: tl.constexpr, TILES_PER_CTA: tl.constexpr):
    pid = tl.program_id(0)
    sensor_id = pid % N_SENSORS
    group = pid // N_SENSORS
    for tile in range(TILES_PER_CTA):
        block = group * TILES_PER_CTA + tile
        ids = block * BLOCK + tl.arange(0, BLOCK)
        valid = ids < n_sources
        _, _, _, _, _, w, i0, alpha = _geometry(
            sensors, sources, intensities, ids, sensor_id, valid,
            delta_bin, r_min, n_bins)
        val0, val1 = (1.0 - alpha) * w, alpha * w
        if STRICT:
            v0, v1 = val0 * fixed_scale, val1 * fixed_scale
            val0 = tl.where(v0 >= 0.0, tl.floor(v0 + 0.5), -tl.floor(-v0 + 0.5)).to(tl.int64)
            val1 = tl.where(v1 >= 0.0, tl.floor(v1 + 0.5), -tl.floor(-v1 + 0.5)).to(tl.int64)
        base = histogram + sensor_id * (n_bins + 1)
        tl.atomic_add(base + i0, val0, mask=valid, sem="relaxed")
        tl.atomic_add(base + i0 + 1, val1, mask=valid, sem="relaxed")


@triton.jit
def _position_backward(sensors, sources, intensities, grad_histogram, partial,
                       n_sources, n_bins, n_blocks, delta_bin, r_min,
                       BLOCK: tl.constexpr):
    block, sensor_id = tl.program_id(0), tl.program_id(1)
    ids = block * BLOCK + tl.arange(0, BLOCK)
    valid = ids < n_sources
    dx, dy, dz, norm, r, w, i0, alpha = _geometry(
        sensors, sources, intensities, ids, sensor_id, valid,
        delta_bin, r_min, n_bins)
    base = grad_histogram + sensor_id * n_bins
    g0 = tl.load(base + i0, mask=valid, other=0.0)
    g1 = tl.load(base + i0 + 1, mask=valid & (i0 + 1 < n_bins), other=0.0)
    interp_grad = (1.0 - alpha) * g0 + alpha * g1
    radial_grad = tl.div_rn(w * (g1 - g0), delta_bin) - tl.div_rn(interp_grad * w, r)
    factor = tl.where(valid & (norm > 0.0), tl.div_rn(radial_grad, norm), 0.0)
    offset = partial + (sensor_id * n_blocks + block) * 3
    tl.store(offset, tl.sum(factor * dx, 0))
    tl.store(offset + 1, tl.sum(factor * dy, 0))
    tl.store(offset + 2, tl.sum(factor * dz, 0))


class _HistogramFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sensors, sources, pc, delta_bin, r_min, n_bins, strict):
        block = 256
        n_blocks = triton.cdiv(pc.numel(), block)
        hist = torch.zeros((sensors.shape[0], n_bins + 1), device=pc.device,
                           dtype=torch.int64 if strict else torch.float32)
        _project[(triton.cdiv(n_blocks, 4) * sensors.shape[0],)](
            sensors, sources, pc, hist, pc.numel(), n_bins, delta_bin, r_min,
            FIXED_POINT_SCALE, STRICT=strict, BLOCK=block,
            N_SENSORS=sensors.shape[0], TILES_PER_CTA=4,
            num_warps=4, enable_fp_fusion=False)
        if strict:
            hist = hist[:, :n_bins].to(torch.float32) * FIXED_POINT_INV_SCALE
        else:
            hist = hist[:, :n_bins]
        ctx.save_for_backward(sensors, sources, pc)
        ctx.delta_bin, ctx.r_min, ctx.n_bins = delta_bin, r_min, n_bins
        return hist

    @staticmethod
    def backward(ctx, grad_hist):
        sensors, sources, pc = ctx.saved_tensors
        block = 256
        n_blocks = triton.cdiv(pc.numel(), block)
        partial = torch.empty((sensors.shape[0], n_blocks, 3),
                              device=pc.device, dtype=torch.float32)
        _position_backward[(n_blocks, sensors.shape[0])](
            sensors, sources, pc, grad_hist.contiguous(), partial, pc.numel(),
            ctx.n_bins, n_blocks, ctx.delta_bin, ctx.r_min, BLOCK=block,
            num_warps=4, enable_fp_fusion=False)
        return partial.sum(dim=1), None, None, None, None, None, None


def kde_histogram_triton(sens_pos_batch, src_pos, Pc, delta_bin=None,
                         r_min=None, n_bins=None, strict_atomics=None):
    """Return ``(B,n_bins)`` soft-bin histograms, with position-only autograd.

    The optional strict flag defaults to the repository environment setting.
    Its forward uses rounded int64 atomics and its backward is the float
    soft-bin STE, matching the existing PyTorch model's derivative semantics.
    Bin indices are clamped before alpha is computed, including edge behavior.
    """
    if src_pos.requires_grad or Pc.requires_grad:
        raise ValueError("Triton histogram supports sensor gradients only; use the PyTorch KDE for source/Pc gradients")
    if sens_pos_batch.ndim != 2 or sens_pos_batch.shape[1] != 3:
        raise ValueError("sens_pos_batch must have shape (B, 3)")
    if src_pos.ndim != 2 or src_pos.shape[1] != 3 or Pc.ndim != 1 or src_pos.shape[0] != Pc.numel():
        raise ValueError("src_pos and Pc must have shapes (K, 3) and (K,)")
    if any(t.device != Pc.device or not t.is_cuda or t.dtype != torch.float32
           for t in (sens_pos_batch, src_pos, Pc)):
        raise ValueError("Triton histogram requires float32 CUDA tensors on the same device")
    if not Pc.numel() or not sens_pos_batch.shape[0]:
        raise ValueError("Triton histogram requires nonempty source and sensor arrays")
    return _HistogramFunction.apply(
        sens_pos_batch.contiguous(), src_pos.contiguous(), Pc.contiguous(),
        KDE_DELTA if delta_bin is None else delta_bin,
        KDE_R_MIN if r_min is None else r_min,
        KDE_N_BINS if n_bins is None else n_bins,
        STRICT_ATOMICS if strict_atomics is None else strict_atomics)
