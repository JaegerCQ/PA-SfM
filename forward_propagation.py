import os
import glob
import math
import argparse

from lib.runtime_config import DEFAULT_SEED, seed_everything

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = seed_everything(DEFAULT_SEED)
STRICT_ATOMICS = os.environ.get("REPRO_STRICT_ATOMICS", "1").strip().lower() not in {"0", "false", "no", "off"}
FIXED_POINT_SCALE = float(os.environ.get("REPRO_FIXED_POINT_SCALE", "10000000000.0"))
FIXED_POINT_INV_SCALE = 1.0 / FIXED_POINT_SCALE

# ===============================================================
# 0. 命令行参数
# ===============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="CUDA 卡号")
parser.add_argument("--sensor-location", type=str, required=True, help="输入探头坐标文件")
parser.add_argument("--ckpt-dir", type=str, required=True, help="checkpoint 根目录")
parser.add_argument("--ckpt-index", type=int, default=-1, help="checkpoint 目录索引，默认 -1 使用最后一个检查点")
parser.add_argument("--output", type=str, required=True, help="输出信号 txt 文件路径")
args = parser.parse_args()

# ===============================================================
# 1. 基础设置与超参数
# ===============================================================
torch.cuda.init()
DEVICE_ID = args.device_id
torch.cuda.set_device(DEVICE_ID)
device = torch.device(f"cuda:{DEVICE_ID}")
print(f"[INIT] 当前运行设备: {device}")

sigma = 0.1e-3
vs = 1500.0

# 40MHz 原采样率，完整 4096 时间窗
delta_time = 25e-9
t_start_idx = 0
n_time_sub = 4096

GRID_SIZE = 400
voxel_size = sigma
center = (GRID_SIZE - 1) / 2.0
n_sources = GRID_SIZE ** 3
sel_step = 1

GAUSS_CUTOFF = 5.0

# ===============================================================
# 2. KDE 参数
# ===============================================================
# KDE_DELTA 越小越准，但 KDE_N_BINS 越大；sigma/8 是比较稳的折中
KDE_DELTA = sigma / 8.0
KDE_R_MIN = -0.01

# 先加载探头坐标，用于估计 KDE_R_MAX
print("[INFO] 正在加载探头坐标...")
loc_all = np.loadtxt(args.sensor_location)
sel_idx = np.arange(0, loc_all.shape[0], sel_step)
loc = loc_all[sel_idx]

sensor_abs_max = float(np.max(np.abs(loc)))
source_abs_max = float(center * voxel_size)

# 保守距离上界：sqrt(3) * (sensor_abs_max + source_abs_max)，再加安全余量
KDE_R_MAX = math.sqrt(3.0) * (sensor_abs_max + source_abs_max) + 0.02
KDE_N_BINS = int(math.ceil((KDE_R_MAX - KDE_R_MIN) / KDE_DELTA)) + 1

KDE_KERNEL_RADIUS = int(math.ceil(GAUSS_CUTOFF * sigma / KDE_DELTA))
KDE_KERNEL_WIDTH = 2 * KDE_KERNEL_RADIUS + 1

print(f"[INFO] grid sources = {n_sources} ({GRID_SIZE}^3), step={voxel_size} m")
print(
    f"[INFO] KDE grid: delta={KDE_DELTA:.3e} m, "
    f"r_range=({KDE_R_MIN:.3e}, {KDE_R_MAX:.3e}), "
    f"bins={KDE_N_BINS}, kernel_width={KDE_KERNEL_WIDTH}"
)

# kernel: d * exp(-d^2 / 2sigma^2)
# histogram 里存的是 Pc / (2r)，所以卷积后就是 Pc/(2r)*d*exp(...)
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

# 时间采样点映射到距离轴 u = v * t
t_rel_grid = torch.arange(n_time_sub, device=device, dtype=torch.float32)
u_time = vs * (t_start_idx + t_rel_grid) * delta_time
pos_time = (u_time - KDE_R_MIN) / KDE_DELTA

time_i0 = torch.floor(pos_time).long().clamp(0, KDE_N_BINS - 2).contiguous()
time_beta = (pos_time - time_i0.to(torch.float32)).contiguous()

# 探头坐标 tensor
sens_x = torch.tensor(loc[:, 0], dtype=torch.float32, device=device).contiguous()
sens_y = torch.tensor(loc[:, 1], dtype=torch.float32, device=device).contiguous()
sens_z = torch.tensor(loc[:, 2], dtype=torch.float32, device=device).contiguous()

# ===============================================================
# 3. Triton KDE 投影核
# hist[sensor, bin] += Pc / (2r)，线性 soft-bin
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
    v0 = (1.0 - alpha) * w * fixed_scale
    v1 = alpha * w * fixed_scale

    q0 = tl.where(v0 >= 0.0, tl.floor(v0 + 0.5), -tl.floor(-v0 + 0.5)).to(tl.int64)
    q1 = tl.where(v1 >= 0.0, tl.floor(v1 + 0.5), -tl.floor(-v1 + 0.5)).to(tl.int64)

    ptr0 = hist_ptr + pid_s * stride_hist_s + i0 * stride_hist_b
    ptr1 = hist_ptr + pid_s * stride_hist_s + (i0 + 1) * stride_hist_b

    tl.atomic_add(ptr0, q0, mask=valid)
    tl.atomic_add(ptr1, q1, mask=valid)


# ===============================================================
# 4. KDE 推理封装
# ===============================================================
def run_forward_inference_kde(Pc, sens_x, sens_y, sens_z):
    n_sensors = sens_x.numel()

    BLOCK_K = 128
    grid = (triton.cdiv(Pc.numel(), BLOCK_K), n_sensors)

    if STRICT_ATOMICS:
        hist_q = torch.zeros(
            (n_sensors, KDE_N_BINS),
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
            num_warps=4,
            num_stages=4,
        )
        hist = hist_q.to(torch.float32) * FIXED_POINT_INV_SCALE
        del hist_q
    else:
        hist = torch.zeros(
            (n_sensors, KDE_N_BINS),
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

    # hist(r) -> p_grid(r)
    # F.conv1d 是 cross-correlation，这里 kernel 不需要翻转
    p_grid = F.conv1d(
        hist[:, None, :],
        KDE_KERNEL,
        padding=KDE_KERNEL_RADIUS,
    )[:, 0, :]

    # p_grid(vt) 线性插值得到 4096 个时间点
    out = (
        (1.0 - time_beta)[None, :] * p_grid[:, time_i0]
        + time_beta[None, :] * p_grid[:, time_i0 + 1]
    ).contiguous()

    return out


# ===============================================================
# 5. 主程序：加载权重并推理
# ===============================================================
if __name__ == "__main__":
    print(f"[INFO] 使用探头数={sens_x.numel()} (每 {sel_step} 个取一个)")
    print("[INFO] KDE 版本不再显式生成 src_x/src_y/src_z 三个 400^3 源坐标数组。")
    print(f"[REPRO] strict_atomics={STRICT_ATOMICS}, fixed_point_scale={FIXED_POINT_SCALE:.3e}")

    # --- 自动寻找 Checkpoint ---
    root_ckpt_dir = args.ckpt_dir
    ckpt_dirs = sorted(
        p for p in glob.glob(os.path.join(root_ckpt_dir, "epoch*"))
        if "seed" not in os.path.basename(p).lower() and "1013" not in os.path.basename(p)
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"在 {root_ckpt_dir} 目录下没有找到任何 checkpoint 文件夹！")

    if abs(args.ckpt_index) > len(ckpt_dirs) or args.ckpt_index >= len(ckpt_dirs):
        raise IndexError(
            f"checkpoint 索引 {args.ckpt_index} 越界，"
            f"当前仅找到 {len(ckpt_dirs)} 个 checkpoint 文件夹。"
        )

    latest_ckpt_dir = ckpt_dirs[args.ckpt_index]
    ckpt_path = os.path.join(latest_ckpt_dir, "model.pt")
    print(f"[INFO] 找到 Checkpoint: {ckpt_path}")

    # --- 加载 Pc 参数 ---
    ckpt = torch.load(ckpt_path, map_location=device)
    Pc = ckpt["Pc_state"].to(device).contiguous()
    print(f"[INFO] 模型参数已加载，所属 Epoch: {ckpt.get('epoch', 'Unknown')}")
    print(f"[INFO] Pc shape={tuple(Pc.shape)}, expected={n_sources}")

    if Pc.numel() != n_sources:
        raise ValueError(
            f"Pc_state 元素数不匹配：got {Pc.numel()}, expected {n_sources}。"
        )

    # --- 执行推理生成信号 ---
    print("[INFO] 开始生成完整探头信号...")
    print(f"[INFO] 设定参数：点数={n_time_sub}, 采样间隔={delta_time * 1e9} ns...")
    print("[INFO] 使用 KDE distance histogram + conv1d 推理。")

    with torch.no_grad():
        simulated_signals = run_forward_inference_kde(Pc, sens_x, sens_y, sens_z)

    simulated_signals_np = simulated_signals.cpu().numpy()
    print(f"[INFO] 信号生成完毕！最终形状为: {simulated_signals_np.shape} (N_sensors x Time_steps)")

    # --- 保存结果到 TXT 文件 ---
    output_filename = args.output
    print(f"[INFO] 正在将结果保存至 {output_filename} ... (文件较大，需稍候)")
    np.savetxt(output_filename, simulated_signals_np, fmt="%.6e")

    print("[SUCCESS] 全量 4096 点探头信号已成功保存！")
