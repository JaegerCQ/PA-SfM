"""Experimental exact-support pruning for the existing 5-sigma KDE forward.

Keep the full histogram and original 128-source arithmetic. The time window
only masks irrelevant histogram deposits or skips provably irrelevant blocks;
it does not change the acoustic kernel or its cutoff.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kde_project_kernel_windowed(
    Pc_ptr, sens_x_ptr, sens_y_ptr, sens_z_ptr, hist_ptr,
    n_sources, n_bins, r_min, delta_r, voxel_size, center,
    stride_hist_s, stride_hist_b, fixed_scale,
    ACTIVE_LO: tl.constexpr, ACTIVE_HI: tl.constexpr,
    BLOCK_SKIP: tl.constexpr, PAIR_MASK: tl.constexpr,
    BLOCK_K: tl.constexpr, GRID_SIZE: tl.constexpr, N_SENSORS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid // N_SENSORS
    pid_s = pid % N_SENSORS
    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    if BLOCK_SKIP:
        # A consecutive flattened block can cross rows. Endpoint min/max on z
        # alone is unsafe for, e.g., source indices 384..511 at GRID_SIZE=400.
        first = pid_k * BLOCK_K
        last = tl.minimum(first + BLOCK_K - 1, n_sources - 1)
        grid2 = GRID_SIZE * GRID_SIZE
        x0, x1 = first // grid2, last // grid2
        rem0, rem1 = first - x0 * grid2, last - x1 * grid2
        y0, y1 = rem0 // GRID_SIZE, rem1 // GRID_SIZE
        z0, z1 = rem0 - y0 * GRID_SIZE, rem1 - y1 * GRID_SIZE
        same_plane = x0 == x1
        same_row = same_plane & (y0 == y1)
        ylo = tl.where(same_plane, y0, 0)
        yhi = tl.where(same_plane, y1, GRID_SIZE - 1)
        zlo = tl.where(same_row, z0, 0)
        zhi = tl.where(same_row, z1, GRID_SIZE - 1)
        pxlo = (x0.to(tl.float32) - center) * voxel_size
        pxhi = (x1.to(tl.float32) - center) * voxel_size
        pylo = (ylo.to(tl.float32) - center) * voxel_size
        pyhi = (yhi.to(tl.float32) - center) * voxel_size
        pzlo = (zlo.to(tl.float32) - center) * voxel_size
        pzhi = (zhi.to(tl.float32) - center) * voxel_size

        dx_min = tl.maximum(tl.maximum(pxlo - sx, sx - pxhi), 0.0)
        dy_min = tl.maximum(tl.maximum(pylo - sy, sy - pyhi), 0.0)
        dz_min = tl.maximum(tl.maximum(pzlo - sz, sz - pzhi), 0.0)
        dx_max = tl.maximum(tl.abs(sx - pxlo), tl.abs(sx - pxhi))
        dy_max = tl.maximum(tl.abs(sy - pylo), tl.abs(sy - pyhi))
        dz_max = tl.maximum(tl.abs(sz - pzlo), tl.abs(sz - pzhi))
        r2_min = dx_min * dx_min + dy_min * dy_min + dz_min * dz_min
        r2_max = dx_max * dx_max + dy_max * dy_max + dz_max * dz_max + 1.0e-12
        # A source deposits at floor(pos) and floor(pos)+1. Four extra bins
        # cover FP boundary error/FMA differences without changing the result.
        r_keep_lo = tl.maximum(0.0, r_min + (ACTIVE_LO - 1 - 4) * delta_r)
        r_keep_hi = r_min + (ACTIVE_HI + 1 + 4) * delta_r
        keep_block = (r2_max >= r_keep_lo * r_keep_lo) & (r2_min <= r_keep_hi * r_keep_hi)
    else:
        keep_block = True

    if keep_block:
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
        if PAIR_MASK:
            valid0 = valid & (i0 >= ACTIVE_LO) & (i0 <= ACTIVE_HI)
            valid1 = valid & (i0 + 1 >= ACTIVE_LO) & (i0 + 1 <= ACTIVE_HI)
        else:
            valid0, valid1 = valid, valid
        tl.atomic_add(ptr0, q0, mask=valid0)
        tl.atomic_add(ptr1, q1, mask=valid1)


def project_windowed_into(pc, sens_x, sens_y, sens_z, histogram, *,
                          grid_size=400, voxel_size, center, r_min, delta_r,
                          fixed_scale, active_lo, active_hi, mode="pair_mask"):
    """Accumulate into a caller-zeroed full-size int64 histogram.

    Modes: pair_mask, block_skip, block_skip_pair_mask. The caller derives the
    inclusive active-bin bounds from its actual interpolation indices and the
    existing Gaussian kernel radius, including both interpolation neighbors.
    """
    if mode not in {"pair_mask", "block_skip", "block_skip_pair_mask"}:
        raise ValueError("Unsupported time-window projection mode")
    if histogram.dtype != torch.int64 or histogram.ndim != 2:
        raise ValueError("histogram must have shape (sensors,bins) and dtype int64")
    n_sensors, n_bins = histogram.shape
    if n_sensors != sens_x.numel() or not 0 <= active_lo <= active_hi < n_bins:
        raise ValueError("Invalid sensor count or active histogram bounds")
    grid = (triton.cdiv(pc.numel(), 128) * n_sensors,)
    kde_project_kernel_windowed[grid](
        pc, sens_x, sens_y, sens_z, histogram, pc.numel(), n_bins,
        r_min, delta_r, voxel_size, center, *histogram.stride(), fixed_scale,
        ACTIVE_LO=active_lo, ACTIVE_HI=active_hi,
        BLOCK_SKIP=mode != "pair_mask", PAIR_MASK=mode != "block_skip",
        BLOCK_K=128, GRID_SIZE=grid_size, N_SENSORS=n_sensors,
        num_warps=4, num_stages=4,
    )
    return histogram
