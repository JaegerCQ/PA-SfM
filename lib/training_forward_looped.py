"""Strict-int64 sensor-fast projection with four source tiles per GPU program.

Keep the original 128-wide arithmetic and per-contribution int64 quantization.
Only program grouping and, optionally, atomic memory ordering change.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def kde_project_kernel_looped(
    Pc_ptr, sens_x_ptr, sens_y_ptr, sens_z_ptr, hist_ptr,
    n_sources, n_bins, r_min, delta_r, voxel_size, center,
    stride_hist_s, stride_hist_b, fixed_scale,
    BLOCK_K: tl.constexpr, GRID_SIZE: tl.constexpr,
    N_SENSORS: tl.constexpr, TILES_PER_CTA: tl.constexpr,
    RELAXED: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_group = pid // N_SENSORS
    pid_s = pid % N_SENSORS
    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    # A regular loop retains the 128-wide vector. It does not concatenate
    # sources into a larger vector or sum floating-point contributions.
    for tile in range(TILES_PER_CTA):
        pid_k = pid_group * TILES_PER_CTA + tile
        k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < n_sources
        pc = tl.load(Pc_ptr + k_offsets, mask=mask_k, other=0.0)
        grid2 = GRID_SIZE * GRID_SIZE
        ix = k_offsets // grid2
        rem = k_offsets - ix * grid2
        iy = rem // GRID_SIZE
        iz = rem - iy * GRID_SIZE
        px = (ix.to(tl.float32) - center) * voxel_size
        py = (iy.to(tl.float32) - center) * voxel_size
        pz = (iz.to(tl.float32) - center) * voxel_size
        dx = sx - px
        dy = sy - py
        dz = sz - pz
        r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)
        pos = (r - r_min) / delta_r
        i0 = pos.to(tl.int32)
        alpha = pos - i0.to(tl.float32)
        valid = mask_k & (i0 >= 0) & (i0 < n_bins - 1)
        w = pc / (2.0 * r)
        v0 = (1.0 - alpha) * w * fixed_scale
        v1 = alpha * w * fixed_scale
        q0 = tl.where(v0 >= 0.0, tl.floor(v0 + 0.5), -tl.floor(-v0 + 0.5)).to(tl.int64)
        q1 = tl.where(v1 >= 0.0, tl.floor(v1 + 0.5), -tl.floor(-v1 + 0.5)).to(tl.int64)
        ptr0 = hist_ptr + pid_s * stride_hist_s + i0 * stride_hist_b
        ptr1 = hist_ptr + pid_s * stride_hist_s + (i0 + 1) * stride_hist_b
        if RELAXED:
            # Only accumulate here; no thread consumes intermediate values or
            # atomic return values. Subsequent kernels run on the same stream.
            # Keep GPU scope so additions across programs remain atomic.
            tl.atomic_add(ptr0, q0, mask=valid, sem="relaxed")
            tl.atomic_add(ptr1, q1, mask=valid, sem="relaxed")
        else:
            tl.atomic_add(ptr0, q0, mask=valid)
            tl.atomic_add(ptr1, q1, mask=valid)


def project_looped_into(pc, sens_x, sens_y, sens_z, histogram, *,
                        grid_size=400, voxel_size, center, r_min, delta_r,
                        fixed_scale, tiles_per_cta=4, relaxed=False):
    """Add to a caller-zeroed full-size histogram, preserving all source pairs."""
    backend = os.environ.get("TRAIN_FORWARD_PROJECTOR", "auto").strip().lower()
    if backend not in {"auto", "shared", "triton"}:
        raise ValueError("TRAIN_FORWARD_PROJECTOR must be auto, shared or triton")
    supported = (pc.is_cuda and grid_size % 16 == 0
                 and torch.cuda.get_device_capability(pc.device) == (8, 0))
    if backend == "shared" and not supported:
        raise ValueError("Shared projection requires an SM80 GPU and grid_size divisible by 16")
    if backend != "triton" and supported:
        from .training_forward_shared import project_looped_into as project_shared
        return project_shared(
            pc, sens_x, sens_y, sens_z, histogram, grid_size=grid_size,
            voxel_size=voxel_size, center=center, r_min=r_min, delta_r=delta_r,
            fixed_scale=fixed_scale, tiles_per_cta=tiles_per_cta, relaxed=relaxed)
    if tiles_per_cta < 1:
        raise ValueError("tiles_per_cta must be positive")
    n_sensors, n_bins = histogram.shape
    grid = (triton.cdiv(pc.numel(), 128 * tiles_per_cta) * n_sensors,)
    return kde_project_kernel_looped[grid](
        pc, sens_x, sens_y, sens_z, histogram, pc.numel(), n_bins,
        r_min, delta_r, voxel_size, center, *histogram.stride(), fixed_scale,
        BLOCK_K=128, GRID_SIZE=grid_size, N_SENSORS=n_sensors,
        TILES_PER_CTA=tiles_per_cta, RELAXED=relaxed,
        num_warps=4, num_stages=4,
    )
