import argparse
from pathlib import Path

from lib.runtime_config import DEFAULT_SEED, seed_everything

import numpy as np

SEED = seed_everything(DEFAULT_SEED)


def rigid_transform_3D(A, B):
    """
    计算从点云 A 到点云 B 的刚体变换 (R, t)
    使得: B ≈ R @ A.T + t
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"矩阵的列数必须为3，当前为 {num_cols}")

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("检测到反射，正在校正旋转矩阵...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T

    return R, t


def build_default_paths(
    pose_id,
    target_pose_id,
    group_id,
    freq_tag,
    ref_file,
    previous_file,
    current_file,
    output_file,
):
    prev_pose_id = pose_id - 1
    group_tag = f"group{group_id}"
    group_pad_tag = f"group{group_id:02d}"

    file_A = Path(ref_file) if ref_file else Path("data") / f"sensor_location_{group_pad_tag}_pose000.txt"

    if previous_file:
        file_B = Path(previous_file)
    else:
        previous_transformed = Path(
            f"transformed_locations_{group_tag}_pose{prev_pose_id}_to_pose_of_{target_pose_id}.txt"
        )

        previous_finetuned = Path(
            f"predicted_locations_{group_tag}_pose{prev_pose_id}_{freq_tag}_"
            f"baseon{prev_pose_id - 1}_corrected_finetuned_masked_{freq_tag}.txt"
        )

        previous_direct_base = Path(
            f"predicted_locations_{group_tag}_pose{prev_pose_id}_{freq_tag}_"
            f"baseon{target_pose_id}_corrected_finetuned_masked_{freq_tag}.txt"
        )

        if previous_transformed.exists():
            file_B = previous_transformed
        elif previous_direct_base.exists():
            file_B = previous_direct_base
        elif previous_finetuned.exists():
            file_B = previous_finetuned
        else:
            file_B = previous_transformed

    if current_file:
        file_C = Path(current_file)
    else:
        file_C = Path(
            f"predicted_locations_{group_tag}_pose{pose_id}_{freq_tag}_"
            f"baseon{prev_pose_id}_corrected_finetuned_masked_{freq_tag}.txt"
        )

    if output_file:
        file_C_new = Path(output_file)
    else:
        file_C_new = Path(f"transformed_locations_{group_tag}_pose{pose_id}_to_pose_of_{target_pose_id}.txt")

    return file_A, file_B, file_C, file_C_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-id", type=int, required=True, help="当前 pose ID，例如 56")
    parser.add_argument("--target-pose-id", type=int, default=0, help="统一到哪个 pose 坐标系，默认 0")
    parser.add_argument("--group-id", type=int, default=3, help="group ID，默认 3")
    parser.add_argument("--freq-tag", type=str, default="10MHz", help="频率标签，默认 10MHz")

    parser.add_argument("--ref-file", type=str, default=None, help="A 坐标文件，默认 data/sensor_location_groupXX_pose000.txt")
    parser.add_argument("--previous-file", type=str, default=None, help="B 坐标文件，不传则自动推断前一帧")
    parser.add_argument("--current-file", type=str, default=None, help="C 坐标文件，不传则自动推断当前帧 refined 文件")
    parser.add_argument("--output", type=str, default=None, help="输出文件，不传则自动生成 transformed 文件名")

    args = parser.parse_args()

    if args.pose_id <= args.target_pose_id:
        raise ValueError(
            f"pose-id={args.pose_id} 必须大于 target-pose-id={args.target_pose_id}。"
        )

    file_A, file_B, file_C, file_C_new = build_default_paths(
        pose_id=args.pose_id,
        target_pose_id=args.target_pose_id,
        group_id=args.group_id,
        freq_tag=args.freq_tag,
        ref_file=args.ref_file,
        previous_file=args.previous_file,
        current_file=args.current_file,
        output_file=args.output,
    )

    print("=" * 80)
    print("[CONFIG]")
    print(f"pose_id={args.pose_id}")
    print(f"target_pose_id={args.target_pose_id}")
    print(f"A/ref file: {file_A}")
    print(f"B/previous pose in target frame: {file_B}")
    print(f"C/current refined file: {file_C}")
    print(f"output: {file_C_new}")
    print("=" * 80)

    for path, name in [(file_A, "A/ref"), (file_B, "B/previous"), (file_C, "C/current")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} 文件不存在: {path}")

    print("正在加载数据...")
    A = np.loadtxt(file_A)
    B = np.loadtxt(file_B)
    C = np.loadtxt(file_C)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    if C.ndim == 1:
        C = C.reshape(1, -1)

    assert A.shape == B.shape == C.shape, (
        f"A, B, C 的形状必须完全一致，当前 A={A.shape}, B={B.shape}, C={C.shape}"
    )

    print("正在计算 A 到 B 的刚体变换 (R, t)...")
    R, t = rigid_transform_3D(A, B)

    A_transformed = (R @ A.T).T + t.T
    rmse = np.sqrt(np.mean((A_transformed - B) ** 2))
    print(f"A 变换到 B 后的 RMSE 误差: {rmse:.6f}")

    print("正在将相同的变换应用到 C...")
    C_new = (R @ C.T).T + t.T

    print(f"正在保存新的 C 坐标到 {file_C_new} ...")
    np.savetxt(file_C_new, C_new, fmt="%.6f", delimiter=" ")

    print("完成！")


if __name__ == "__main__":
    main()
