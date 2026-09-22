"""Optional fixed-point training adjoint tiled over sensors and sources.

Every pair is rounded to int64 before the sensor reduction. This preserves the
reference summation semantics. The default four-sensor specialization was
checked against the reference on A100 with Triton 3.1.0. Revalidate numerical
equivalence with the accompanying benchmarks when changing GPU/compiler.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kde_project_backward_sensor_tiled(
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    grad_hist_ptr, grad_pc_ptr,
    n_sources, n_sensors, n_bins,
    r_min, delta_r, voxel_size, center,
    stride_grad_hist_s, stride_grad_hist_b, fixed_scale,
    BLOCK_K: tl.constexpr, BLOCK_S: tl.constexpr, GRID_SIZE: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_k = k_offsets < n_sources
    mask_s = s_offsets < n_sensors

    grid2 = GRID_SIZE * GRID_SIZE
    ix = k_offsets // grid2
    rem = k_offsets - ix * grid2
    iy = rem // GRID_SIZE
    iz = rem - iy * GRID_SIZE

    px = (ix.to(tl.float32) - center) * voxel_size
    py = (iy.to(tl.float32) - center) * voxel_size
    pz = (iz.to(tl.float32) - center) * voxel_size

    sx = tl.load(sens_x_ptr + s_offsets, mask=mask_s, other=0.0)
    sy = tl.load(sens_y_ptr + s_offsets, mask=mask_s, other=0.0)
    sz = tl.load(sens_z_ptr + s_offsets, mask=mask_s, other=0.0)

    dx = sx[:, None] - px[None, :]
    dy = sy[:, None] - py[None, :]
    dz = sz[:, None] - pz[None, :]

    # Keep the reference expression and default fusion policy. The benchmark
    # rejects specializations whose compiler changes per-pair rounding.
    r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)
    pos = (r - r_min) / delta_r
    i0 = pos.to(tl.int32)
    alpha = pos - i0.to(tl.float32)
    valid = mask_k[None, :] & mask_s[:, None] & (i0 >= 0) & (i0 < n_bins - 1)

    ptr0 = grad_hist_ptr + s_offsets[:, None] * stride_grad_hist_s + i0 * stride_grad_hist_b
    ptr1 = grad_hist_ptr + s_offsets[:, None] * stride_grad_hist_s + (i0 + 1) * stride_grad_hist_b
    g0 = tl.load(ptr0, mask=valid, other=0.0)
    g1 = tl.load(ptr1, mask=valid, other=0.0)

    # Reference BLOCK_K=128 PTX rounds (1-alpha)*g0, then fuses alpha*g1
    # with it. For larger tensors LLVM otherwise selects the opposite FMA
    # operand, changing the subsequently quantized int64 value.
    grad_k = tl.fma(alpha, g1, (1.0 - alpha) * g0) / (2.0 * r)
    v = grad_k * fixed_scale
    q = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(-v + 0.5)).to(tl.int64)
    q = tl.where(valid, q, 0)
    q_sum = tl.sum(q, axis=0)
    tl.atomic_add(grad_pc_ptr + k_offsets, q_sum, mask=mask_k)


def project_backward_fixed_tiled(sens_x, sens_y, sens_z, grad_hist, n_sources,
                                 n_bins, r_min, delta_r, voxel_size, center,
                                 fixed_scale=1.0e10, grid_size=400,
                                 block_s=4, num_warps=4, output=None):
    """Launch the integer adjoint and return its int64 sums.

    ``output`` must be zero-initialized by the caller; omitting it allocates a
    zeroed array. No float conversion is done here so equality can be checked
    before potentially masking integer discrepancies through float rounding.
    """
    if block_s not in (1, 4, 8, 16, 32, 64):
        raise ValueError("block_s must be one of 1, 4, 8, 16, 32, 64")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if output is None:
        output = torch.zeros(n_sources, device=grad_hist.device, dtype=torch.int64)
    kde_project_backward_sensor_tiled[(triton.cdiv(n_sources, 128), triton.cdiv(sens_x.numel(), block_s))](
        sens_x, sens_y, sens_z, grad_hist, output,
        n_sources, sens_x.numel(), n_bins, r_min, delta_r, voxel_size, center,
        grad_hist.stride(0), grad_hist.stride(1), fixed_scale,
        BLOCK_K=128, BLOCK_S=block_s, GRID_SIZE=grid_size,
        num_warps=num_warps, num_stages=4,
    )
    return output


def project_gradient_tiled(sens_x, sens_y, sens_z, grad_hist, n_sources,
                           n_bins, r_min, delta_r, voxel_size, center,
                           fixed_scale=1.0e10, grid_size=400, block_k=128,
                           block_s=4, num_warps=4):
    """Training integration entry point; returns fixed-point int64 gradients."""
    if block_k != 128:
        raise ValueError("The tiled adjoint preserves the reference BLOCK_K=128")
    return project_backward_fixed_tiled(
        sens_x, sens_y, sens_z, grad_hist, n_sources, n_bins, r_min, delta_r,
        voxel_size, center, fixed_scale=fixed_scale, grid_size=grid_size,
        block_s=block_s, num_warps=num_warps)
