import os
import time
import math
import argparse

from lib.runtime_config import (
    DEFAULT_SEED,
    reproducible_run_tag,
    seed_everything,
    torch_generator,
)
from lib.time_scatter import scatter_time_gradient, time_scatter_indices_unique

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = seed_everything(DEFAULT_SEED)
STRICT_ATOMICS = os.environ.get("REPRO_STRICT_ATOMICS", "1").strip().lower() not in {"0", "false", "no", "off"}
FIXED_POINT_SCALE = float(os.environ.get("REPRO_FIXED_POINT_SCALE", "10000000000.0"))
FIXED_POINT_INV_SCALE = 1.0 / FIXED_POINT_SCALE
TRAIN_TIME_SCATTER = os.environ.get("TRAIN_TIME_SCATTER", "vectorized").strip().lower()
if TRAIN_TIME_SCATTER not in {"legacy", "vectorized"}:
    raise ValueError("TRAIN_TIME_SCATTER must be legacy or vectorized")
TRAIN_FORWARD_LAYOUT = os.environ.get("TRAIN_FORWARD_LAYOUT", "sensor_fast").strip().lower()
if TRAIN_FORWARD_LAYOUT not in {"legacy", "sensor_fast"}:
    raise ValueError("TRAIN_FORWARD_LAYOUT must be legacy or sensor_fast")
if TRAIN_FORWARD_LAYOUT == "sensor_fast" and not STRICT_ATOMICS:
    raise ValueError("TRAIN_FORWARD_LAYOUT=sensor_fast requires REPRO_STRICT_ATOMICS=1")
TRAIN_BACKWARD_BACKEND = os.environ.get("TRAIN_BACKWARD_BACKEND", "sensor_tiled").strip().lower()
if TRAIN_BACKWARD_BACKEND not in {"legacy", "sensor_tiled"}:
    raise ValueError("TRAIN_BACKWARD_BACKEND must be legacy or sensor_tiled")
if TRAIN_BACKWARD_BACKEND == "sensor_tiled":
    if not STRICT_ATOMICS:
        raise ValueError("TRAIN_BACKWARD_BACKEND=sensor_tiled requires REPRO_STRICT_ATOMICS=1")
    from lib.training_backward_tiled import project_gradient_tiled

# ===============================================================
# 命令行参数
# ===============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="CUDA 卡号")
parser.add_argument("--sensor-location", type=str, required=True, help="输入探头坐标文件")
parser.add_argument("--signal", type=str, required=True, help="输入信号文件")
parser.add_argument("--save-dir", type=str, required=True, help="checkpoint 保存根目录")
args = parser.parse_args()

# ===============================================================
# CUDA 初始化
# ===============================================================
torch.cuda.init()
DEVICE_ID = args.device_id
torch.cuda.set_device(DEVICE_ID)
device = torch.device(f"cuda:{DEVICE_ID}")
print(f"[INIT] 当前运行设备: {device}")
print(f"[CONFIG] TRAIN_FORWARD_LAYOUT={TRAIN_FORWARD_LAYOUT}")
print(f"[CONFIG] TRAIN_BACKWARD_BACKEND={TRAIN_BACKWARD_BACKEND}")

# ===============================================================
# 声学参数 / 分辨率
# ===============================================================
sigma = 0.1e-3
vs = 1500.0
delta_time = 100e-9

# ===============================================================
# 仿真时间范围
# ===============================================================
t_start_idx = 500
t_end_idx = 1000

# ===============================================================
# 探头位置 & 参考信号
# ===============================================================
loc_all = np.loadtxt(args.sensor_location)
signals_all = np.loadtxt(args.signal) / 50

signals_all = signals_all[:, ::4]

n_sensors_full, n_time = signals_all.shape
n_time_sub = t_end_idx - t_start_idx
print(
    f"[INFO] sensors={n_sensors_full}, full time={n_time}, "
    f"sim time range=({t_start_idx}, {t_end_idx}), len={n_time_sub}"
)

sel_step = 1
sel_idx = np.arange(0, loc_all.shape[0], sel_step)
loc = loc_all[sel_idx]
signals = signals_all[sel_idx][:, t_start_idx:t_end_idx]
n_sensors = loc.shape[0]
print(f"[INFO] 使用探头数={n_sensors} (每 {sel_step} 个取一个)")

sens_x = torch.tensor(loc[:, 0], dtype=torch.float32, device=device).contiguous()
sens_y = torch.tensor(loc[:, 1], dtype=torch.float32, device=device).contiguous()
sens_z = torch.tensor(loc[:, 2], dtype=torch.float32, device=device).contiguous()

# ===============================================================
# 源点网格
# ===============================================================
GRID_SIZE = 400
voxel_size = sigma
center = (GRID_SIZE - 1) / 2.0
n_sources = GRID_SIZE ** 3

print(f"[INFO] grid sources = {n_sources} ({GRID_SIZE}^3), step={voxel_size} m")

src_x = None
src_y = None
src_z = None

Pc_param = torch.nn.Parameter(
    torch.randn(n_sources, device=device, generator=torch_generator(device=device, seed=SEED))
)

# ===============================================================
# KDE 参数
# ===============================================================
GAUSS_CUTOFF = 5.0

KDE_DELTA = sigma / 8.0
KDE_R_MIN = -0.01

sensor_abs_max = float(np.max(np.abs(loc)))
source_abs_max = float(center * voxel_size)

KDE_R_MAX = math.sqrt(3.0) * (sensor_abs_max + source_abs_max) + 0.02
KDE_N_BINS = int(math.ceil((KDE_R_MAX - KDE_R_MIN) / KDE_DELTA)) + 1

KDE_KERNEL_RADIUS = int(math.ceil(GAUSS_CUTOFF * sigma / KDE_DELTA))
KDE_KERNEL_WIDTH = 2 * KDE_KERNEL_RADIUS + 1

print(
    f"[INFO] KDE grid: delta={KDE_DELTA:.3e} m, "
    f"r_range=({KDE_R_MIN:.3e}, {KDE_R_MAX:.3e}), "
    f"bins={KDE_N_BINS}, kernel_width={KDE_KERNEL_WIDTH}"
)

d_grid = (
    torch.arange(
        -KDE_KERNEL_RADIUS,
        KDE_KERNEL_RADIUS + 1,
        device=device,
        dtype=torch.float32,
    )
    * KDE_DELTA
)

KDE_KERNEL = (
    d_grid * torch.exp(-(d_grid * d_grid) / (2.0 * sigma * sigma))
).view(1, 1, -1).contiguous()

t_rel_grid = torch.arange(n_time_sub, device=device, dtype=torch.float32)
u_time = vs * (t_start_idx + t_rel_grid) * delta_time
pos_time = (u_time - KDE_R_MIN) / KDE_DELTA

time_i0 = torch.floor(pos_time).long().clamp(0, KDE_N_BINS - 2).contiguous()
time_beta = (pos_time - time_i0.to(torch.float32)).contiguous()
TIME_SCATTER_UNIQUE = (
    time_scatter_indices_unique(time_i0) if TRAIN_TIME_SCATTER == "vectorized" else None
)
if TRAIN_TIME_SCATTER == "vectorized" and not TIME_SCATTER_UNIQUE:
    print("[CONFIG] TRAIN_TIME_SCATTER=vectorized, effective=legacy (interpolation destinations overlap)")
else:
    print(f"[CONFIG] TRAIN_TIME_SCATTER={TRAIN_TIME_SCATTER}, effective={TRAIN_TIME_SCATTER}")

# ===============================================================
# Triton 前向核：source -> distance KDE histogram
# hist[sensor, bin] += Pc / (2r)
# ===============================================================
@triton.jit
def kde_project_kernel(
    Pc_ptr,
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    hist_ptr,
    n_sources,
    n_bins,
    r_min,
    delta_r,
    voxel_size,
    center,
    stride_hist_s,
    stride_hist_b,
    BLOCK_K: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)

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

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    dx = sx - px
    dy = sy - py
    dz = sz - pz

    r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)

    pos = (r - r_min) / delta_r
    i0 = pos.to(tl.int32)
    alpha = pos - i0.to(tl.float32)

    valid = mask_k & (i0 >= 0) & (i0 < n_bins - 1)

    w = pc / (2.0 * r)

    ptr0 = hist_ptr + pid_s * stride_hist_s + i0 * stride_hist_b
    ptr1 = hist_ptr + pid_s * stride_hist_s + (i0 + 1) * stride_hist_b

    tl.atomic_add(ptr0, (1.0 - alpha) * w, mask=valid)
    tl.atomic_add(ptr1, alpha * w, mask=valid)


@triton.jit
def kde_project_kernel_fixed(
    Pc_ptr,
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    hist_ptr,
    n_sources,
    n_bins,
    r_min,
    delta_r,
    voxel_size,
    center,
    stride_hist_s,
    stride_hist_b,
    fixed_scale,
    BLOCK_K: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    SENSOR_FAST: tl.constexpr = False,
    N_SENSORS: tl.constexpr = 1,
):
    if SENSOR_FAST:
        pid = tl.program_id(0)
        pid_k = pid // N_SENSORS
        pid_s = pid % N_SENSORS
    else:
        pid_k = tl.program_id(0)
        pid_s = tl.program_id(1)

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

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

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

    tl.atomic_add(ptr0, q0, mask=valid)
    tl.atomic_add(ptr1, q1, mask=valid)


# ===============================================================
# Triton 反向核：grad_hist -> grad_pc
# ===============================================================
@triton.jit
def kde_project_backward_kernel(
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    grad_hist_ptr,
    grad_pc_ptr,
    n_sources,
    n_bins,
    r_min,
    delta_r,
    voxel_size,
    center,
    stride_grad_hist_s,
    stride_grad_hist_b,
    BLOCK_K: tl.constexpr,
    GRID_SIZE: tl.constexpr,
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

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    dx = sx - px
    dy = sy - py
    dz = sz - pz

    r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)

    pos = (r - r_min) / delta_r
    i0 = pos.to(tl.int32)
    alpha = pos - i0.to(tl.float32)

    valid = mask_k & (i0 >= 0) & (i0 < n_bins - 1)

    ptr0 = grad_hist_ptr + pid_s * stride_grad_hist_s + i0 * stride_grad_hist_b
    ptr1 = grad_hist_ptr + pid_s * stride_grad_hist_s + (i0 + 1) * stride_grad_hist_b

    g0 = tl.load(ptr0, mask=valid, other=0.0)
    g1 = tl.load(ptr1, mask=valid, other=0.0)

    grad_k = ((1.0 - alpha) * g0 + alpha * g1) / (2.0 * r)

    tl.atomic_add(grad_pc_ptr + k_offsets, grad_k, mask=valid)


@triton.jit
def kde_project_backward_kernel_fixed(
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    grad_hist_ptr,
    grad_pc_ptr,
    n_sources,
    n_bins,
    r_min,
    delta_r,
    voxel_size,
    center,
    stride_grad_hist_s,
    stride_grad_hist_b,
    fixed_scale,
    BLOCK_K: tl.constexpr,
    GRID_SIZE: tl.constexpr,
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

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    dx = sx - px
    dy = sy - py
    dz = sz - pz

    r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-12)

    pos = (r - r_min) / delta_r
    i0 = pos.to(tl.int32)
    alpha = pos - i0.to(tl.float32)

    valid = mask_k & (i0 >= 0) & (i0 < n_bins - 1)

    ptr0 = grad_hist_ptr + pid_s * stride_grad_hist_s + i0 * stride_grad_hist_b
    ptr1 = grad_hist_ptr + pid_s * stride_grad_hist_s + (i0 + 1) * stride_grad_hist_b

    g0 = tl.load(ptr0, mask=valid, other=0.0)
    g1 = tl.load(ptr1, mask=valid, other=0.0)

    grad_k = ((1.0 - alpha) * g0 + alpha * g1) / (2.0 * r)
    v = grad_k * fixed_scale
    q = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(-v + 0.5)).to(tl.int64)

    tl.atomic_add(grad_pc_ptr + k_offsets, q, mask=valid)


# ===============================================================
# Autograd Function
# ===============================================================
class GaussianSimFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Pc, src_x, src_y, src_z,
        sens_x, sens_y, sens_z,
        t_start_idx, n_time_sub,
        delta_t, vs, sigma,
    ):
        BLOCK_K = 128

        n_source_blocks = triton.cdiv(Pc.numel(), BLOCK_K)
        sensor_fast = TRAIN_FORWARD_LAYOUT == "sensor_fast"
        # Flatten the launch: swapping the original axes would exceed grid.y's
        # limit with 500000 source blocks. Only the strict forward path changes.
        grid = (n_source_blocks * sens_x.numel(),) if sensor_fast else (n_source_blocks, sens_x.numel())

        if STRICT_ATOMICS:
            hist_q = torch.zeros(
                (sens_x.numel(), KDE_N_BINS),
                device=Pc.device,
                dtype=torch.int64,
            )
            kde_project_kernel_fixed[grid](
                Pc,
                sens_x, sens_y, sens_z,
                hist_q,
                Pc.numel(),
                KDE_N_BINS,
                KDE_R_MIN,
                KDE_DELTA,
                voxel_size,
                center,
                hist_q.stride(0),
                hist_q.stride(1),
                FIXED_POINT_SCALE,
                BLOCK_K=BLOCK_K,
                GRID_SIZE=GRID_SIZE,
                SENSOR_FAST=sensor_fast,
                N_SENSORS=sens_x.numel(),
                num_warps=4,
                num_stages=4,
            )
            hist = hist_q.to(torch.float32) * FIXED_POINT_INV_SCALE
            del hist_q
        else:
            hist = torch.zeros(
                (sens_x.numel(), KDE_N_BINS),
                device=Pc.device,
                dtype=torch.float32,
            )
            kde_project_kernel[grid](
                Pc,
                sens_x, sens_y, sens_z,
                hist,
                Pc.numel(),
                KDE_N_BINS,
                KDE_R_MIN,
                KDE_DELTA,
                voxel_size,
                center,
                hist.stride(0),
                hist.stride(1),
                BLOCK_K=BLOCK_K,
                GRID_SIZE=GRID_SIZE,
                num_warps=4,
                num_stages=4,
            )

        p_grid = F.conv1d(
            hist[:, None, :],
            KDE_KERNEL,
            padding=KDE_KERNEL_RADIUS,
        )[:, 0, :]

        y = (
            (1.0 - time_beta)[None, :] * p_grid[:, time_i0]
            + time_beta[None, :] * p_grid[:, time_i0 + 1]
        ).contiguous()

        ctx.save_for_backward(sens_x, sens_y, sens_z, time_i0, time_beta)
        ctx.n_sources = Pc.numel()
        ctx.n_sensors = sens_x.numel()
        ctx.n_bins = KDE_N_BINS

        return y

    @staticmethod
    def backward(ctx, grad_out):
        sens_x, sens_y, sens_z, saved_time_i0, saved_time_beta = ctx.saved_tensors

        grad_p_grid = scatter_time_gradient(
            grad_out, saved_time_i0, saved_time_beta, ctx.n_bins,
            mode=TRAIN_TIME_SCATTER, indices_unique=TIME_SCATTER_UNIQUE,
        )

        grad_hist = F.conv_transpose1d(
            grad_p_grid[:, None, :],
            KDE_KERNEL,
            padding=KDE_KERNEL_RADIUS,
        )[:, 0, :].contiguous()

        BLOCK_K = 128
        grid = (triton.cdiv(ctx.n_sources, BLOCK_K), ctx.n_sensors)

        if STRICT_ATOMICS and TRAIN_BACKWARD_BACKEND == "sensor_tiled":
            grad_pc_q = project_gradient_tiled(
                sens_x, sens_y, sens_z, grad_hist,
                ctx.n_sources, ctx.n_bins, KDE_R_MIN, KDE_DELTA,
                voxel_size, center, fixed_scale=FIXED_POINT_SCALE,
                grid_size=GRID_SIZE,
            )
            grad_pc = grad_pc_q.to(torch.float32) * FIXED_POINT_INV_SCALE
            del grad_pc_q
        elif STRICT_ATOMICS:
            grad_pc_q = torch.zeros(
                ctx.n_sources,
                device=grad_out.device,
                dtype=torch.int64,
            )
            kde_project_backward_kernel_fixed[grid](
                sens_x, sens_y, sens_z,
                grad_hist,
                grad_pc_q,
                ctx.n_sources,
                ctx.n_bins,
                KDE_R_MIN,
                KDE_DELTA,
                voxel_size,
                center,
                grad_hist.stride(0),
                grad_hist.stride(1),
                FIXED_POINT_SCALE,
                BLOCK_K=BLOCK_K,
                GRID_SIZE=GRID_SIZE,
                num_warps=4,
                num_stages=4,
            )
            grad_pc = grad_pc_q.to(torch.float32) * FIXED_POINT_INV_SCALE
            del grad_pc_q
        else:
            grad_pc = torch.zeros(
                ctx.n_sources,
                device=grad_out.device,
                dtype=torch.float32,
            )
            kde_project_backward_kernel[grid](
                sens_x, sens_y, sens_z,
                grad_hist,
                grad_pc,
                ctx.n_sources,
                ctx.n_bins,
                KDE_R_MIN,
                KDE_DELTA,
                voxel_size,
                center,
                grad_hist.stride(0),
                grad_hist.stride(1),
                BLOCK_K=BLOCK_K,
                GRID_SIZE=GRID_SIZE,
                num_warps=4,
                num_stages=4,
            )

        return grad_pc, None, None, None, None, None, None, None, None, None, None, None


def gaussian_sim(
    Pc, src_x, src_y, src_z,
    sens_x, sens_y, sens_z,
    t_start_idx, n_time_sub,
    delta_t, vs, sigma,
):
    return GaussianSimFunction.apply(
        Pc, src_x, src_y, src_z,
        sens_x, sens_y, sens_z,
        t_start_idx, n_time_sub,
        delta_t, vs, sigma,
    )


# ===============================================================
# TGV 正则
# ===============================================================
def forward_diff(tensor, dim):
    diff = torch.roll(tensor, shifts=-1, dims=dim) - tensor
    idx = [slice(None)] * tensor.ndim
    idx[dim] = slice(-1, None)
    diff[tuple(idx)] = 0.0
    return diff


def tgv2_regularization(A_flat, grid_shape, alpha0=2.0, alpha1=1.0, eps=1e-8):
    A = A_flat.view(*grid_shape)

    gx = forward_diff(A, 0)
    gy = forward_diff(A, 1)
    gz = forward_diff(A, 2)

    grad_mag = torch.sqrt(gx * gx + gy * gy + gz * gz + eps)
    first_term = grad_mag.mean()

    gxx = forward_diff(gx, 0)
    gyy = forward_diff(gy, 1)
    gzz = forward_diff(gz, 2)

    gxy = 0.5 * (forward_diff(gx, 1) + forward_diff(gy, 0))
    gxz = 0.5 * (forward_diff(gx, 2) + forward_diff(gz, 0))
    gyz = 0.5 * (forward_diff(gy, 2) + forward_diff(gz, 1))

    second_sq = (
        gxx * gxx + gyy * gyy + gzz * gzz
        + 2.0 * (gxy * gxy + gxz * gxz + gyz * gyz)
    )
    second_term = torch.sqrt(second_sq + eps).mean()

    return alpha1 * first_term + alpha0 * second_term


GRID_SHAPE = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
lambda_tv = 8.0
tgv_alpha0 = 2.0
tgv_alpha1 = 1.0

# ===============================================================
# 训练循环
# ===============================================================
if __name__ == "__main__":
    print("\n[Train] 开始训练循环 (KDE accelerated Gaussian)...")
    print(f"[REPRO] strict_atomics={STRICT_ATOMICS}, fixed_point_scale={FIXED_POINT_SCALE:.3e}")

    target = torch.tensor(signals, dtype=torch.float32, device=device).contiguous()

    optimizer = torch.optim.Adam([Pc_param], lr=0.5)

    max_epochs = 100
    root_ckpt_dir = args.save_dir
    os.makedirs(root_ckpt_dir, exist_ok=True)

    for epoch in range(max_epochs):
        t_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        Pc = F.softplus(Pc_param)

        y_pred = gaussian_sim(
            Pc, src_x, src_y, src_z,
            sens_x, sens_y, sens_z,
            t_start_idx, n_time_sub,
            delta_time, vs, sigma,
        )

        data_loss = torch.mean((y_pred - target) ** 2)
        tv_loss = tgv2_regularization(
            Pc,
            GRID_SHAPE,
            alpha0=tgv_alpha0,
            alpha1=tgv_alpha1,
        )

        loss = data_loss + lambda_tv * tv_loss

        loss.backward()
        optimizer.step()

        grad_mean = Pc_param.grad.mean().item() if Pc_param.grad is not None else 0.0
        # The scalar read synchronizes preceding CUDA work. Time after it so
        # asynchronous launches are not reported as near-zero training time.
        epoch_time = time.time() - t_start

        print(
            f"[Epoch {epoch + 1}/{max_epochs}] "
            f"loss={loss.item():.6e}, "
            f"data={data_loss.item():.6e}, "
            f"tv={tv_loss.item():.6e}, "
            f"grad_mean={grad_mean:.4e}, "
            f"time={epoch_time:.2f}s, "
            f"Pc_min={Pc.min().item():.3e}, "
            f"Pc_mean={Pc.mean().item():.3e}"
        )

        if (epoch + 1) % 50 == 0:
            ckpt_dir = os.path.join(root_ckpt_dir, f"epoch{epoch + 1:03d}_{reproducible_run_tag(SEED)}")
            os.makedirs(ckpt_dir, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "Pc_state": Pc.detach().cpu(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss.item(),
                    "epoch_time": 0.0,
                    "kde_delta": KDE_DELTA,
                    "kde_r_min": KDE_R_MIN,
                    "kde_r_max": KDE_R_MAX,
                    "kde_n_bins": KDE_N_BINS,
                },
                os.path.join(ckpt_dir, "model.pt"),
            )

            print(f"  [Checkpoint] 模型已保存到: {ckpt_dir}")

    print("[Train] 完成。")
