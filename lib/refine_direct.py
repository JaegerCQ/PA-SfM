"""Experimental continuous-Gaussian forward and analytic sensor backward.

No KDE, interpolation, fixed-point quantization, or floating-point atomics.
Per-tile sums are reduced in a fixed order by PyTorch. Different reduction
grouping can differ from the reference in float32, and is measured explicitly.
Only first-order sensor-position gradients are supported; source positions,
source amplitudes, time samples and sigma are fixed. Double backward is rejected.
"""

import math
import numbers
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _geometry(sensors, sources, pc_ptr, sensor, ids, valid):
    dx = tl.load(sensors + sensor * 3) - tl.load(sources + ids * 3, valid, 0.0)
    dy = tl.load(sensors + sensor * 3 + 1) - tl.load(sources + ids * 3 + 1, valid, 0.0)
    dz = tl.load(sensors + sensor * 3 + 2) - tl.load(sources + ids * 3 + 2, valid, 0.0)
    # Match torch.norm's three-component CUDA reduction order.
    norm = libdevice.sqrt_rn((dx * dx + dz * dz) + dy * dy)
    r = norm + 1.0e-12
    pc = tl.load(pc_ptr + ids, valid, 0.0)
    return dx, dy, dz, norm, r, pc


@triton.jit
def _forward(sensors, sources, pc_ptr, ct_ptr, partial,
             K: tl.constexpr, T: tl.constexpr, NB: tl.constexpr,
             INV_2SIGMA2: tl.constexpr, SUPPORT: tl.constexpr,
             BK: tl.constexpr, BT: tl.constexpr):
    kb, tb, sensor = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ids = kb * BK + tl.arange(0, BK)
    times = tb * BT + tl.arange(0, BT)
    valid = ids < K
    _, _, _, _, r, pc = _geometry(sensors, sources, pc_ptr, sensor, ids, valid)
    ct = tl.load(ct_ptr + times, times < T, 0.0)
    ct_first = tl.min(tl.where(times < T, ct, float("inf")), axis=0)
    ct_last = tl.max(tl.where(times < T, ct, -float("inf")), axis=0)
    r_lo = tl.min(tl.where(valid, r, float("inf")), axis=0)
    r_hi = tl.max(tl.where(valid, r, -float("inf")), axis=0)
    # Monotonic float subtraction gives the minimum/maximum possible rt.
    # Skip only if every source/time exponential is guaranteed float32 zero.
    if (r_lo - ct_last <= SUPPORT) & (r_hi - ct_first >= -SUPPORT):
        rt = r[:, None] - ct[None, :]
        # At 16 sigma, exp(-128) is below half the smallest float32
        # subnormal. This excludes exact-zero exponentials, not Gaussian tails.
        live = valid[:, None] & (times[None, :] < T) & (tl.abs(rt) <= SUPPORT)
        exponent = libdevice.exp(-(rt * rt) * INV_2SIGMA2)
        amplitude = (pc[:, None] * 0.5) * tl.div_rn(rt, r[:, None])
        value = tl.where(live, amplitude * exponent, 0.0)
        tl.store(partial + (sensor * NB + kb) * T + times,
                 tl.sum(value, axis=0), mask=times < T)
    else:
        tl.store(partial + (sensor * NB + kb) * T + times, 0.0, mask=times < T)


@triton.jit
def _backward(sensors, sources, pc_ptr, ct_ptr, grad_output, partial,
              K: tl.constexpr, T: tl.constexpr, NB: tl.constexpr, NT: tl.constexpr,
              INV_2SIGMA2: tl.constexpr, INV_SIGMA2: tl.constexpr,
              SUPPORT: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr):
    kb, tb, sensor = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ids = kb * BK + tl.arange(0, BK)
    times = tb * BT + tl.arange(0, BT)
    valid = ids < K
    dx, dy, dz, norm, r, pc = _geometry(sensors, sources, pc_ptr, sensor, ids, valid)
    ct = tl.load(ct_ptr + times, times < T, 0.0)
    go = tl.load(grad_output + sensor * T + times, times < T, 0.0)
    ct_first = tl.min(tl.where(times < T, ct, float("inf")), axis=0)
    ct_last = tl.max(tl.where(times < T, ct, -float("inf")), axis=0)
    r_lo = tl.min(tl.where(valid, r, float("inf")), axis=0)
    r_hi = tl.max(tl.where(valid, r, -float("inf")), axis=0)
    # Monotonic float subtraction gives the minimum/maximum possible rt.
    # Skip only if every source/time exponential is guaranteed float32 zero.
    if (r_lo - ct_last <= SUPPORT) & (r_hi - ct_first >= -SUPPORT):
        rt = r[:, None] - ct[None, :]
        live = valid[:, None] & (times[None, :] < T) & (tl.abs(rt) <= SUPPORT)
        exponent = libdevice.exp(-(rt * rt) * INV_2SIGMA2)
        # dF/dr = Pc/2 * exp * [ct/r^2 - (r-ct)^2/(r*sigma^2)].
        derivative = ((pc[:, None] * 0.5) * exponent
                      * (tl.div_rn(ct[None, :], r[:, None] * r[:, None])
                         - tl.div_rn(rt * rt, r[:, None]) * INV_SIGMA2))
        radial = tl.sum(tl.where(live, derivative * go[None, :], 0.0), axis=1)
        factor = tl.where(valid & (norm > 0.0), tl.div_rn(radial, norm), 0.0)
        base = ((sensor * NB + kb) * NT + tb) * 3
        tl.store(partial + base, tl.sum(factor * dx, axis=0))
        tl.store(partial + base + 1, tl.sum(factor * dy, axis=0))
        tl.store(partial + base + 2, tl.sum(factor * dz, axis=0))
    else:
        base = ((sensor * NB + kb) * NT + tb) * 3
        tl.store(partial + base, 0.0)
        tl.store(partial + base + 1, 0.0)
        tl.store(partial + base + 2, 0.0)


class _DirectGaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sensors, sources, pc, ct, sigma):
        bk, bt = 128, 32
        k, t, b = pc.numel(), ct.numel(), sensors.shape[0]
        nb, nt = triton.cdiv(k, bk), triton.cdiv(t, bt)
        partial = torch.empty((b, nb, t), dtype=torch.float32, device=pc.device)
        _forward[(nb, nt, b)](
            sensors, sources, pc, ct, partial, k, t, nb,
            1.0 / (2.0 * sigma * sigma), 16.0 * sigma, bk, bt,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(sensors, sources, pc, ct)
        ctx.sigma, ctx.bk, ctx.bt = sigma, bk, bt
        return partial.sum(dim=1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        sensors, sources, pc, ct = ctx.saved_tensors
        bk, bt, sigma = ctx.bk, ctx.bt, ctx.sigma
        k, t, b = pc.numel(), ct.numel(), sensors.shape[0]
        nb, nt = triton.cdiv(k, bk), triton.cdiv(t, bt)
        partial = torch.empty((b, nb, nt, 3), device=pc.device, dtype=torch.float32)
        _backward[(nb, nt, b)](
            sensors, sources, pc, ct, grad_output.contiguous(), partial,
            k, t, nb, nt, 1.0 / (2.0 * sigma * sigma), 1.0 / (sigma * sigma),
            16.0 * sigma, bk, bt, num_warps=4, enable_fp_fusion=False)
        return partial.sum(dim=(1, 2)), None, None, None, None


def direct_gaussian(sensors, sources, pc, ct, sigma=1.0e-4):
    """Continuous direct simulation with first-order sensor gradients only.

    Inputs are dense float32 CUDA tensors on one device: sensors (B,3), sources
    (K,3), pc (K,), and ct (T,), with B/K/T positive. Inputs must be finite.
    ct stores distance=velocity*time and can have arbitrary sample ordering;
    construct it using the reference float32 arithmetic when comparing outputs.
    sigma is a fixed, positive finite real scalar. Source/Pc/ct gradients and
    second-order sensor gradients are intentionally unsupported.
    """
    tensors = (sensors, sources, pc, ct)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("sensors, sources, pc and ct must be torch tensors")
    if sensors.ndim != 2 or sensors.shape[1] != 3 or sensors.shape[0] == 0:
        raise ValueError("sensors must have nonempty shape (B,3)")
    if sources.ndim != 2 or sources.shape[1] != 3 or sources.shape[0] == 0:
        raise ValueError("sources must have nonempty shape (K,3)")
    if pc.ndim != 1 or pc.numel() != sources.shape[0]:
        raise ValueError("pc must have shape (K,) matching sources")
    if ct.ndim != 1 or ct.numel() == 0:
        raise ValueError("ct must have nonempty shape (T,)")
    if not isinstance(sigma, numbers.Real) or isinstance(sigma, bool):
        raise TypeError("sigma must be a fixed positive finite real scalar")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    # Kernel constants must remain representable in float32, even for an
    # otherwise finite Python scalar. This also avoids Python underflow in a*a.
    square = sigma * sigma
    if square == 0.0 or not math.isfinite(square):
        raise ValueError("sigma squared is outside the supported numeric range")
    inverse_square = 1.0 / square
    if inverse_square > torch.finfo(torch.float32).max or inverse_square < torch.finfo(torch.float32).tiny:
        raise ValueError("inverse sigma squared must be a normal finite float32 value")
    if any(t.dtype != torch.float32 or not t.is_cuda or t.layout != torch.strided for t in tensors):
        raise ValueError("dense CUDA float32 inputs required")
    if any(t.device != sensors.device for t in tensors):
        raise ValueError("all input tensors must be on the same CUDA device")
    if sources.requires_grad or pc.requires_grad or ct.requires_grad:
        raise ValueError("Only first-order sensor-position gradients are implemented")
    return _DirectGaussian.apply(sensors.contiguous(), sources.contiguous(), pc.contiguous(),
                                 ct.contiguous(), sigma)
