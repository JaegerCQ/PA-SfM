"""Experimental multi-sensor tiles for the strict-int64 training forward model.

The production training path does not import this module. The source block stays
at 128; all acoustic operations and per-contribution quantization are retained.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kde_project_tiled_kernel_fixed(
    Pc_ptr,
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    hist_ptr,
    n_sources,
    n_sensors,
    n_bins,
    r_min,
    delta_r,
    voxel_size,
    center,
    stride_hist_shard,
    stride_hist_s,
    stride_hist_b,
    fixed_scale,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    SENSOR_GROUPS: tl.constexpr,
    HIST_SHARDS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid // SENSOR_GROUPS
    sensor_group = pid % SENSOR_GROUPS
    sensor_ids = sensor_group * BLOCK_S + tl.arange(0, BLOCK_S)
    sensor_mask = sensor_ids < n_sensors
    shard = pid_k % HIST_SHARDS

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

    sx = tl.load(sens_x_ptr + sensor_ids, mask=sensor_mask, other=0.0)
    sy = tl.load(sens_y_ptr + sensor_ids, mask=sensor_mask, other=0.0)
    sz = tl.load(sens_z_ptr + sensor_ids, mask=sensor_mask, other=0.0)

    dx = sx[:, None] - px[None, :]
    dy = sy[:, None] - py[None, :]
    dz = sz[:, None] - pz[None, :]

    r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)
    pos = (r - r_min) / delta_r
    i0 = pos.to(tl.int32)
    alpha = pos - i0.to(tl.float32)
    valid = sensor_mask[:, None] & mask_k[None, :] & (i0 >= 0) & (i0 < n_bins - 1)

    w = pc[None, :] / (2.0 * r)
    v0 = (1.0 - alpha) * w * fixed_scale
    v1 = alpha * w * fixed_scale
    q0 = tl.where(v0 >= 0.0, tl.floor(v0 + 0.5), -tl.floor(-v0 + 0.5)).to(tl.int64)
    q1 = tl.where(v1 >= 0.0, tl.floor(v1 + 0.5), -tl.floor(-v1 + 0.5)).to(tl.int64)

    ptr0 = hist_ptr + shard * stride_hist_shard + sensor_ids[:, None] * stride_hist_s + i0 * stride_hist_b
    ptr1 = hist_ptr + shard * stride_hist_shard + sensor_ids[:, None] * stride_hist_s + (i0 + 1) * stride_hist_b
    tl.atomic_add(ptr0, q0, mask=valid)
    tl.atomic_add(ptr1, q1, mask=valid)


def project_tiled_into(
    pc, sens_x, sens_y, sens_z, histogram,
    *, grid_size, voxel_size, center, r_min, delta_r, fixed_scale,
    block_sensors=4, num_warps=4,
):
    """Clear a preallocated [shards,sensors,bins] buffer, project, merge int64.

    The returned [sensors,bins] tensor aliases ``histogram`` when there is one
    shard. Multiple shards are combined using integer addition, after original
    per-contribution quantization; floating-point histogram reduction is avoided.
    """
    if histogram.dtype != torch.int64 or histogram.ndim != 3:
        raise ValueError("histogram must be an int64 [shards,sensors,bins] tensor")
    if block_sensors not in {1, 2, 4, 8} or num_warps not in {4, 8}:
        raise ValueError("supported sensor tiles are 1/2/4/8 and warps are 4/8")
    shards, n_sensors, n_bins = histogram.shape
    if shards not in {1, 2, 4} or sens_x.numel() != n_sensors:
        raise ValueError("histogram shape does not match sensors or supported shard count")
    histogram.zero_()
    sensor_groups = triton.cdiv(n_sensors, block_sensors)
    grid = (triton.cdiv(pc.numel(), 128) * sensor_groups,)
    kde_project_tiled_kernel_fixed[grid](
        pc, sens_x, sens_y, sens_z, histogram,
        pc.numel(), n_sensors, n_bins, r_min, delta_r, voxel_size, center,
        *histogram.stride(), fixed_scale,
        BLOCK_K=128, BLOCK_S=block_sensors, GRID_SIZE=grid_size,
        SENSOR_GROUPS=sensor_groups, HIST_SHARDS=shards,
        num_warps=num_warps, num_stages=4,
    )
    return histogram[0] if shards == 1 else histogram.sum(dim=0)
