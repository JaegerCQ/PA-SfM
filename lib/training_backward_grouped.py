"""BS4 adjoint with deferred integer reduction across sensor lanes.

Every source/sensor pair keeps the original floating expression and quantizer.
Each lane accumulates quantized int64 contributions across sensor groups, and
lanes are summed only once after the loop. This removes repeated warp shuffles
without changing integer addition modulo 2**64. API, launch settings and group
support remain unchanged: the training wrapper selects BS4/G256/BK128/W4 for
1024 sensors, while partial-sensor programs still add to a zeroed output.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kde_project_backward_sensor_grouped(
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    grad_hist_ptr, grad_pc_ptr,
    n_sources, n_sensors, n_bins,
    r_min, delta_r, voxel_size, center,
    stride_grad_hist_s, stride_grad_hist_b, fixed_scale,
    BLOCK_K: tl.constexpr, BLOCK_S: tl.constexpr, GRID_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr, OWNS_ALL_SENSORS: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < n_sources

    grid2 = GRID_SIZE * GRID_SIZE
    ix = k_offsets // grid2
    rem = k_offsets - ix * grid2
    iy = rem // GRID_SIZE
    iz = rem - iy * GRID_SIZE

    px = (ix.to(tl.float32) - center) * voxel_size
    py = (iy.to(tl.float32) - center) * voxel_size
    pz = (iz.to(tl.float32) - center) * voxel_size

    accumulated_q = tl.full((BLOCK_S, BLOCK_K), 0, tl.int64)
    for group in tl.range(0, GROUPS_PER_PROGRAM):
        s_offsets = (pid_s * GROUPS_PER_PROGRAM + group) * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = s_offsets < n_sensors
        sx = tl.load(sens_x_ptr + s_offsets, mask=mask_s, other=0.0)
        sy = tl.load(sens_y_ptr + s_offsets, mask=mask_s, other=0.0)
        sz = tl.load(sens_z_ptr + s_offsets, mask=mask_s, other=0.0)

        dx = sx[:, None] - px[None, :]
        dy = sy[:, None] - py[None, :]
        dz = sz[:, None] - pz[None, :]

        r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)
        pos = (r - r_min) / delta_r
        i0 = pos.to(tl.int32)
        alpha = pos - i0.to(tl.float32)
        valid = mask_k[None, :] & mask_s[:, None] & (i0 >= 0) & (i0 < n_bins - 1)

        ptr0 = grad_hist_ptr + s_offsets[:, None] * stride_grad_hist_s + i0 * stride_grad_hist_b
        ptr1 = grad_hist_ptr + s_offsets[:, None] * stride_grad_hist_s + (i0 + 1) * stride_grad_hist_b
        g0 = tl.load(ptr0, mask=valid, other=0.0)
        g1 = tl.load(ptr1, mask=valid, other=0.0)

        grad_k = tl.fma(alpha, g1, (1.0 - alpha) * g0) / (2.0 * r)
        v = grad_k * fixed_scale
        q = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(-v + 0.5)).to(tl.int64)
        q = tl.where(valid, q, 0)
        accumulated_q = accumulated_q + q

    accumulated_q = tl.sum(accumulated_q, axis=0)
    if OWNS_ALL_SENSORS:
        tl.store(grad_pc_ptr + k_offsets, accumulated_q, mask=mask_k)
    else:
        tl.atomic_add(grad_pc_ptr + k_offsets, accumulated_q, mask=mask_k, sem="relaxed")


def project_backward_fixed_tiled(sens_x, sens_y, sens_z, grad_hist, n_sources,
                                 n_bins, r_min, delta_r, voxel_size, center,
                                 fixed_scale=1.0e10, grid_size=400,
                                 block_s=4, num_warps=4, output=None,
                                 groups_per_program=16):
    """Return int64 gradients; partial-sensor output must start at zero.

    With exclusive source ownership the output is overwritten, including when
    the caller supplies it. Training explicitly selects 256 groups; the helper
    also supports partial-sensor groupings with zero-initialized output.
    """
    if block_s != 4 or num_warps != 4:
        raise ValueError("This experiment fixes BLOCK_S=4 and num_warps=4")
    if groups_per_program not in (1, 4, 16, 64, 256):
        raise ValueError("groups_per_program must be one of 1, 4, 16, 64, 256")
    n_sensor_programs = triton.cdiv(sens_x.numel(), block_s * groups_per_program)
    owns_all_sensors = n_sensor_programs == 1
    if output is None:
        allocate = torch.empty if owns_all_sensors else torch.zeros
        output = allocate(n_sources, device=grad_hist.device, dtype=torch.int64)
    kde_project_backward_sensor_grouped[(triton.cdiv(n_sources, 128), n_sensor_programs)](
        sens_x, sens_y, sens_z, grad_hist, output,
        n_sources, sens_x.numel(), n_bins, r_min, delta_r, voxel_size, center,
        grad_hist.stride(0), grad_hist.stride(1), fixed_scale,
        BLOCK_K=128, BLOCK_S=block_s, GRID_SIZE=grid_size,
        GROUPS_PER_PROGRAM=groups_per_program, OWNS_ALL_SENSORS=owns_all_sensors,
        num_warps=num_warps, num_stages=4,
    )
    return output


def project_gradient_tiled(sens_x, sens_y, sens_z, grad_hist, n_sources,
                           n_bins, r_min, delta_r, voxel_size, center,
                           fixed_scale=1.0e10, grid_size=400, block_k=128,
                           block_s=4, num_warps=4, groups_per_program=16):
    """Training integration interface with an explicit sensor grouping size."""
    if block_k != 128:
        raise ValueError("This experiment preserves BLOCK_K=128")
    return project_backward_fixed_tiled(
        sens_x, sens_y, sens_z, grad_hist, n_sources, n_bins, r_min, delta_r,
        voxel_size, center, fixed_scale=fixed_scale, grid_size=grid_size,
        block_s=block_s, num_warps=num_warps, groups_per_program=groups_per_program)
