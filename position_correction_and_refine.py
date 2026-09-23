import os
import time
import math
import argparse
import re
from pathlib import Path

from lib.runtime_config import DEFAULT_SEED, seed_everything

import numpy as np
import torch
import torch.nn as nn

from lib.refine_direct import direct_gaussian

SEED = seed_everything(DEFAULT_SEED)
REFINE_BACKEND = os.environ.get("REFINE_BACKEND", "triton").strip().lower()
if REFINE_BACKEND not in {"triton", "torch"}:
    raise ValueError("REFINE_BACKEND must be 'triton' or 'torch'")

# ===============================================================
# 命令行参数
# ===============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--pose-id", type=int, required=True, help="当前 pose ID，例如 56")
parser.add_argument("--ckpt-dir", type=str, required=True, help="用于 refine 的 checkpoint 目录，例如 checkpoints_07")
parser.add_argument("--device-id", type=int, default=0, help="CUDA 卡号，默认 0")
parser.add_argument("--sensor-location", type=str, required=True, help="参考探头坐标文件")
parser.add_argument("--base-pose-id", type=int, default=None, help="当前定位基于哪个 base pose；默认 pose-id - 1")
parser.add_argument("--num-localization-parts", type=int, default=1, help="localization 分片数量")
args = parser.parse_args()

# ===============================================================
# 基础路径配置
# ===============================================================
POSE_ID = args.pose_id
PREV_POSE_ID = args.base_pose_id if args.base_pose_id is not None else POSE_ID - 1
GROUP_ID = 3
FREQ_TAG = "10MHz"
DEVICE_ID = args.device_id
REFINE_CKPT_DIR = Path(args.ckpt_dir)
NUM_LOCALIZATION_PARTS = args.num_localization_parts

GROUP_TAG = f"group{GROUP_ID}"
GROUP_PAD_TAG = f"group{GROUP_ID:02d}"
POSE_TAG = f"pose{POSE_ID}"
BASE_TAG = f"baseon{PREV_POSE_ID}"

in_dir = Path(f"outputs_{POSE_ID}")
merged_path = Path(f"predicted_locations_{GROUP_TAG}_{POSE_TAG}_{FREQ_TAG}_{BASE_TAG}.txt")
corrected_path = Path(f"predicted_locations_{GROUP_TAG}_{POSE_TAG}_{FREQ_TAG}_{BASE_TAG}_corrected.txt")
finetuned_path = Path(f"predicted_locations_{GROUP_TAG}_{POSE_TAG}_{FREQ_TAG}_{BASE_TAG}_corrected_finetuned_masked_{FREQ_TAG}.txt")
loc_ref_path = Path(args.sensor_location)
signal_path = Path(f"simulated_signals_full_4096_{GROUP_TAG}_{POSE_TAG}.txt")

print("=" * 80)
print("[CONFIG]")
print(f"POSE_ID={POSE_ID}")
print(f"PREV_POSE_ID={PREV_POSE_ID}")
print(f"input dir={in_dir}")
print(f"merged_path={merged_path}")
print(f"corrected_path={corrected_path}")
print(f"finetuned_path={finetuned_path}")
print(f"loc_ref_path={loc_ref_path}")
print(f"signal_path={signal_path}")
print(f"refine ckpt dir={REFINE_CKPT_DIR}")
print(f"device id={DEVICE_ID}")
print(f"num localization parts={NUM_LOCALIZATION_PARTS}")
print("=" * 80)

# ===============================================================
# 1. 合并 localization 分片
# ===============================================================
part_files = [
    in_dir / f"pred_{POSE_TAG}_part{i}.txt"
    for i in range(NUM_LOCALIZATION_PARTS)
]

merged = None

for f in part_files:
    print(f"[LOAD] {f}")
    arr = np.loadtxt(f)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if merged is None:
        merged = np.full_like(arr, np.nan, dtype=np.float64)
    else:
        if arr.shape != merged.shape:
            raise ValueError(f"Shape mismatch: {f} has {arr.shape}, expected {merged.shape}")

    valid = ~np.isnan(arr[:, 0])

    overlap = valid & ~np.isnan(merged[:, 0])
    if np.any(overlap):
        dup_ids = np.where(overlap)[0]
        raise RuntimeError(f"Duplicate sensor ids found in {f}: {dup_ids[:20]} ...")

    merged[valid] = arr[valid]
    print(f"       valid rows: {valid.sum()}")

missing = np.isnan(merged[:, 0])
n_missing = missing.sum()

print("=" * 80)
print(f"[INFO] merged shape: {merged.shape}")
print(f"[INFO] solved sensors: {(~missing).sum()}")
print(f"[INFO] missing sensors: {n_missing}")

if n_missing > 0:
    missing_ids = np.where(missing)[0]
    print(f"[WARN] missing sensor ids: {missing_ids[:50]}")
    if len(missing_ids) > 50:
        print(f"       ... total missing {len(missing_ids)}")

np.savetxt(
    merged_path,
    merged,
    fmt="%.6f",
    header=f"Predicted x, y, z (meters); merged from {in_dir}/pred_{POSE_TAG}_part*.txt"
)

print(f"[DONE] saved to {merged_path}")

# ===============================================================
# 2. RANSAC 刚体校正
# ===============================================================
RANSAC_RNG = np.random.default_rng(SEED)

def rigid_transform_3D(A, B):
    """普通 SVD 求解 R, t"""
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t.reshape(3, 1)


def get_distances(points):
    """计算 3 个点两两之间的距离"""
    d1 = np.linalg.norm(points[0] - points[1])
    d2 = np.linalg.norm(points[0] - points[2])
    d3 = np.linalg.norm(points[1] - points[2])
    return np.array([d1, d2, d3])


def ransac_rigid_transform_robust(src_points, dst_points, threshold=0.005, iterations=1000000, rng=None):
    """
    针对低内点率优化的 RANSAC
    threshold: 判断内点的距离阈值，例如 0.005m
    iterations: 尝试采样的次数
    """
    n_points = src_points.shape[0]
    best_inliers = []
    best_R = np.eye(3)
    best_t = np.zeros((3, 1))

    dist_tolerance = threshold * 2.0

    print(f"开始鲁棒 RANSAC: 总点数 {n_points}, 目标内点阈值 {threshold}m, 迭代 {iterations} 次")
    start_time = time.time()

    valid_sample_count = 0
    rng = np.random.default_rng(SEED) if rng is None else rng

    for i in range(iterations):
        idx = rng.choice(n_points, 3, replace=False)

        src_sample = src_points[idx]
        dst_sample = dst_points[idx]

        d_src = get_distances(src_sample)
        d_dst = get_distances(dst_sample)

        if np.any(np.abs(d_src - d_dst) > dist_tolerance):
            continue

        valid_sample_count += 1

        R, t = rigid_transform_3D(src_sample, dst_sample)

        src_transformed = (np.dot(R, src_points.T) + t).T
        errors = np.linalg.norm(dst_points - src_transformed, axis=1)

        current_inliers = np.where(errors < threshold)[0]

        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_R = R
            best_t = t
            print(f"  Iter {i}: 发现更好模型 -> 内点数: {len(best_inliers)}")

    print(f"RANSAC 结束: 耗时 {time.time() - start_time:.2f}s, 有效几何采样 {valid_sample_count} 次")
    print(f"最终找到内点数: {len(best_inliers)}")

    if len(best_inliers) >= 3:
        final_R, final_t = rigid_transform_3D(src_points[best_inliers], dst_points[best_inliers])
        return final_R, final_t, best_inliers

    return best_R, best_t, best_inliers


try:
    loc_ref = np.loadtxt(loc_ref_path)
    loc_pred = np.loadtxt(merged_path)
except Exception as e:
    print(f"数据加载失败: {e}")
    raise

if loc_ref.ndim == 1:
    loc_ref = loc_ref.reshape(1, -1)
if loc_pred.ndim == 1:
    loc_pred = loc_pred.reshape(1, -1)

if loc_ref.shape != loc_pred.shape:
    raise ValueError(f"loc_ref 和 loc_pred shape 不一致: {loc_ref.shape} vs {loc_pred.shape}")

# 只使用 localization 有效的探头参与 RANSAC
valid_pred_mask = ~np.isnan(loc_pred[:, 0])
valid_original_ids = np.where(valid_pred_mask)[0]

loc_ref_valid = loc_ref[valid_pred_mask]
loc_pred_valid = loc_pred[valid_pred_mask]

print(f"[INFO] RANSAC 使用有效定位探头数: {loc_ref_valid.shape[0]} / {loc_ref.shape[0]}")

if loc_ref_valid.shape[0] < 3:
    raise RuntimeError(f"有效定位探头少于 3 个，无法估计刚体变换: {loc_ref_valid.shape[0]}")

R_opt, t_opt, inliers_valid = ransac_rigid_transform_robust(
    loc_ref_valid,
    loc_pred_valid,
    threshold=0.005,
    iterations=100000,
    rng=RANSAC_RNG,
)

# 将有效子集里的 inlier 下标映射回 1024 原始探头 ID
inliers = valid_original_ids[inliers_valid]

# 将刚体变换应用到完整 1024 个探头，最终仍输出完整阵列 pose
loc_corrected = (np.dot(R_opt, loc_ref.T) + t_opt).T

# 误差只在有效 localization 探头上计算
loc_corrected_valid = loc_corrected[valid_pred_mask]
diff_final_valid = np.linalg.norm(loc_pred_valid - loc_corrected_valid, axis=1)

print("\n" + "=" * 80)
print("鲁棒算法结果验证")
print("=" * 80)

inliers_sorted = np.sort(inliers)
print(f"算法认定为 Inliers 的原始探头 ID ({len(inliers)}个):\n{inliers_sorted}")

print("-" * 80)
inlier_valid_positions = np.isin(valid_original_ids, inliers_sorted)

if np.any(inlier_valid_positions):
    print(f"内点平均误差: {np.mean(diff_final_valid[inlier_valid_positions]):.6f} m")
else:
    print("[WARN] 没有找到有效内点，无法计算内点平均误差")

np.savetxt(corrected_path, loc_corrected, fmt="%.6f")
print(f"矫正结果已保存: {corrected_path}")

# ===============================================================
# 3. 基于信号的刚体 refine
# ===============================================================
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
print(f"[INIT] 当前运行设备: {device}")

vs = 1500.0

delta_time = 100e-9
t_start_idx = 500
t_end_idx = 1000
n_time_sub = t_end_idx - t_start_idx

TOTAL_EPOCHS = 100
LR_ROT = 1e-4
LR_TRANS = 1e-4
SIGMA_FIXED = 0.1e-3

MICRO_BATCH_SIZE = 4

INCLUDE_IDS = np.asarray(inliers_sorted, dtype=np.int64).tolist()

if len(INCLUDE_IDS) < 3:
    raise RuntimeError(f"refine 可用 inlier 少于 3 个: {len(INCLUDE_IDS)}")

# ===============================================================
# 3.1 数据加载
# ===============================================================
loc_path = corrected_path

ckpt_dir = REFINE_CKPT_DIR
def checkpoint_sort_key(path):
    match = re.search(r"epoch(\d+)", str(path))
    epoch = int(match.group(1)) if match else -1
    return epoch, str(path)


ckpt_files = sorted(
    (
        p for p in ckpt_dir.rglob("*.pt")
        if "seed" not in str(p).lower() and "1013" not in str(p)
    ),
    key=checkpoint_sort_key,
)
if not ckpt_files:
    raise FileNotFoundError(f"在 {ckpt_dir} 下找不到任何 .pt 模型文件")

CKPT_PATH = str(ckpt_files[-1])
KEEP_RATIO = 0.002

if os.path.exists(CKPT_PATH):
    print(f"[LOAD] Model: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    Pc_raw = ckpt["Pc_state"].to(device).contiguous().requires_grad_(False)

    GRID_SIZE = 400
    coords = (np.arange(GRID_SIZE) - (GRID_SIZE - 1) / 2.0) * 0.1e-3
    # Index the same float32 axis table without allocating three 400^3 arrays.
    coords_tensor = torch.tensor(coords, device=device, dtype=torch.float32)

    n_keep = int(Pc_raw.numel() * KEEP_RATIO)
    val_top, idx_top = torch.topk(torch.abs(Pc_raw), n_keep)

    Pc_fixed = Pc_raw[idx_top].contiguous()
    ix = idx_top // (GRID_SIZE * GRID_SIZE)
    iy = (idx_top // GRID_SIZE) % GRID_SIZE
    iz = idx_top % GRID_SIZE
    xyz_src = torch.stack([coords_tensor[ix], coords_tensor[iy], coords_tensor[iz]], dim=1).contiguous()
else:
    raise FileNotFoundError(f"找不到模型文件: {CKPT_PATH}")

signals_full_raw = np.loadtxt(signal_path)
signals_full_raw = signals_full_raw[:, ::4]

sel_step = 1
sel_idx = np.arange(0, signals_full_raw.shape[0], sel_step)
signals_sub_raw = signals_full_raw[sel_idx]

signals_target = torch.tensor(
    signals_sub_raw[:, t_start_idx:t_end_idx],
    dtype=torch.float32,
    device=device
)
NUM_TOTAL_SENSORS = signals_target.shape[0]

init_pos_np = np.loadtxt(loc_path)

if init_pos_np.shape[0] != NUM_TOTAL_SENSORS:
    raise ValueError(f"坐标数量 ({init_pos_np.shape[0]}) 与信号数量 ({NUM_TOTAL_SENSORS}) 不匹配！")

init_pos_tensor = torch.tensor(init_pos_np, dtype=torch.float32, device=device)

active_indices_np = INCLUDE_IDS
active_indices_tensor = torch.tensor(active_indices_np, dtype=torch.long, device=device)

print(f"[CONFIG] Refinement backend: {REFINE_BACKEND}")
print(f"[CONFIG] 总探头数: {NUM_TOTAL_SENSORS}")
print(f"[CONFIG] 信号已降采样，当前截取维度: {signals_target.shape[1]}")
print(f"[CONFIG] 包含探头ID数量: {len(INCLUDE_IDS)}")
print(f"[CONFIG] INCLUDE_IDS 前 50 个: {INCLUDE_IDS[:50]}")

# ===============================================================
# 3.2 核心函数与模型
# ===============================================================
def build_rotation_matrix(r):
    cx, sx = torch.cos(r[0]), torch.sin(r[0])
    cy, sy = torch.cos(r[1]), torch.sin(r[1])
    cz, sz = torch.cos(r[2]), torch.sin(r[2])

    Rx = torch.stack([
        torch.tensor([1., 0., 0.], device=device),
        torch.stack([torch.tensor(0., device=device), cx, -sx]),
        torch.stack([torch.tensor(0., device=device), sx, cx])
    ])

    Ry = torch.stack([
        torch.stack([cy, torch.tensor(0., device=device), sy]),
        torch.tensor([0., 1., 0.], device=device),
        torch.stack([-sy, torch.tensor(0., device=device), cy])
    ])

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.tensor(0., device=device)]),
        torch.stack([sz, cz, torch.tensor(0., device=device)]),
        torch.tensor([0., 0., 1.], device=device)
    ])

    return Rz @ Ry @ Rx


def _reference_acoustic_simulation(sens_pos_batch, src_pos, Pc, t_start, n_time, dt, vs, a):
    diff = sens_pos_batch.unsqueeze(1) - src_pos.unsqueeze(0)
    r = torch.norm(diff, dim=2) + 1e-12

    t_global = t_start + torch.arange(n_time, device=device, dtype=torch.float32)

    rt = r.unsqueeze(2) - vs * (t_global * dt).reshape(1, 1, -1)
    exponent = torch.exp(-(rt ** 2) / (2.0 * a * a))
    amplitude = Pc.reshape(1, -1, 1) * 0.5 * (rt / r.unsqueeze(2))

    signals = torch.sum(amplitude * exponent, dim=1)
    return signals


_DIRECT_CT_CACHE = {}


def vectorized_acoustic_simulation(sens_pos_batch, src_pos, Pc, t_start, n_time, dt, vs, a):
    if REFINE_BACKEND == "torch" or sens_pos_batch.device.type != "cuda":
        return _reference_acoustic_simulation(sens_pos_batch, src_pos, Pc, t_start, n_time, dt, vs, a)
    key = (str(sens_pos_batch.device), t_start, n_time, float(dt), float(vs))
    ct = _DIRECT_CT_CACHE.get(key)
    if ct is None:
        # Preserve the reference's time arithmetic order, not (vs * dt) * t.
        t_global = t_start + torch.arange(n_time, device=sens_pos_batch.device, dtype=torch.float32)
        ct = vs * (t_global * dt)
        _DIRECT_CT_CACHE[key] = ct
    return direct_gaussian(sens_pos_batch, src_pos, Pc, ct, a)


def batch_negative_correlation_loss(pred_batch, target_batch):
    pred_mean = pred_batch - pred_batch.mean(dim=1, keepdim=True)
    target_mean = target_batch - target_batch.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_mean * target_mean, dim=1)
    d1 = torch.norm(pred_mean, dim=1)
    d2 = torch.norm(target_mean, dim=1)
    correlation = numerator / (d1 * d2 + 1e-9)
    return 1.0 - correlation.mean()


class RigidArrayOptimizer(nn.Module):
    def __init__(self, init_points):
        super().__init__()
        self.centroid = torch.mean(init_points, dim=0, keepdim=True).detach()
        self.points_centered = (init_points - self.centroid).detach()

        self.rot_euler = nn.Parameter(torch.zeros(3, device=device))
        self.trans_vec = nn.Parameter(torch.zeros(3, device=device))

    def forward(self):
        R = build_rotation_matrix(self.rot_euler)
        p_final = (self.points_centered @ R.T) + self.centroid + self.trans_vec
        return p_final


# ===============================================================
# 3.3 训练循环
# ===============================================================
print("\n" + "=" * 80)
print(f"开始优化 (Valid Sensors Only, Total: {len(active_indices_np)})")
print("=" * 80)

model = RigidArrayOptimizer(init_pos_tensor).to(device)
optimizer = torch.optim.Adam([
    {"params": model.rot_euler, "lr": LR_ROT},
    {"params": model.trans_vec, "lr": LR_TRANS}
])

loss_history = []
start_time = time.time()

for epoch in range(TOTAL_EPOCHS):
    optimizer.zero_grad()

    current_pos_all = model()

    total_loss_val = 0.0
    num_active = len(active_indices_np)

    for i in range(0, num_active, MICRO_BATCH_SIZE):
        start_k = i
        end_k = min(i + MICRO_BATCH_SIZE, num_active)

        batch_ids = active_indices_tensor[start_k:end_k]

        pos_batch = current_pos_all[batch_ids]
        target_batch = signals_target[batch_ids]

        pred_batch = vectorized_acoustic_simulation(
            pos_batch,
            xyz_src,
            Pc_fixed,
            t_start_idx,
            n_time_sub,
            delta_time,
            vs,
            a=SIGMA_FIXED
        )

        loss_fragment = batch_negative_correlation_loss(pred_batch, target_batch)

        weight = (end_k - start_k) / num_active
        weighted_loss = loss_fragment * weight
        weighted_loss.backward(retain_graph=True)

        total_loss_val += weighted_loss.item()
        del pred_batch, loss_fragment, weighted_loss

    optimizer.step()
    del current_pos_all

    loss_history.append(total_loss_val)

    if epoch % 2 == 0:
        r_deg = model.rot_euler.detach().cpu().numpy() * 180 / np.pi
        t_mm = model.trans_vec.detach().cpu().numpy() * 1000
        print(f"Epoch {epoch:04d} | Active_Loss: {total_loss_val:.6f} | "
              f"dRot(deg): [{r_deg[0]:.4f}, {r_deg[1]:.4f}, {r_deg[2]:.4f}] | "
              f"dTrans(mm): [{t_mm[0]:.4f}, {t_mm[1]:.4f}, {t_mm[2]:.4f}]")

print(f"\n[DONE] 优化耗时: {time.time() - start_time:.1f}s")

# ===============================================================
# 4. 结果保存
# ===============================================================
final_pos_tensor = model().detach()
final_pos_np = final_pos_tensor.cpu().numpy()

init_pos_cpu = init_pos_tensor.cpu().numpy()
if 25 not in INCLUDE_IDS and 25 < NUM_TOTAL_SENSORS:
    check_idx = 25
    move_dist = np.linalg.norm(final_pos_np[check_idx] - init_pos_cpu[check_idx]) * 1000
    print(f"\n[CHECK] 未监督探头 #{check_idx} 的跟随移动距离: {move_dist:.4f} mm")

np.savetxt(
    finetuned_path,
    final_pos_np,
    fmt="%.8f",
    header="Finetuned x, y, z (meters)"
)
print(f"[SAVE] 所有探头坐标(含未监督的)已保存至: {finetuned_path}")
print(f"[SAVE] final shape: {final_pos_np.shape}")
