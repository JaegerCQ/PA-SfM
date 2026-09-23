#!/usr/bin/env bash

set -u
# Resolve all data, Python entry points and outputs inside this package.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CHECK_ONLY=0
case "${1:-}" in
    --check) CHECK_ONLY=1 ;;
    --help|-h)
        echo "Usage: bash run_group3_pose_range.sh [--check]"
        echo "Default: poses 0..9 on one A100 (GPU 0)."
        echo "Example: END_POSE=1 bash run_group3_pose_range.sh"
        exit 0
        ;;
    "") ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
esac
if [ "$#" -gt 1 ]; then
    echo "[ERROR] Expected at most one argument (--check or --help)" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export REPRO_SEED="${REPRO_SEED:-1013}"
export PYTHONHASHSEED="$REPRO_SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export REPRO_DETERMINISTIC="${REPRO_DETERMINISTIC:-1}"
export REPRO_DETERMINISTIC_WARN_ONLY="${REPRO_DETERMINISTIC_WARN_ONLY:-1}"
export REPRO_ALLOW_TF32="${REPRO_ALLOW_TF32:-1}"
export REPRO_STRICT_ATOMICS="${REPRO_STRICT_ATOMICS:-1}"
export REPRO_FIXED_POINT_SCALE="${REPRO_FIXED_POINT_SCALE:-10000000000.0}"
export REFINE_BACKEND="${REFINE_BACKEND:-triton}"
export LOCALIZATION_FINE_EXECUTION="${LOCALIZATION_FINE_EXECUTION:-auto}"
export LOCALIZATION_PROJECTOR="${LOCALIZATION_PROJECTOR:-triton}"
export TRAIN_TIME_SCATTER="${TRAIN_TIME_SCATTER:-vectorized}"
export TRAIN_FORWARD_PROJECTOR="${TRAIN_FORWARD_PROJECTOR:-auto}"
export TRAIN_FORWARD_LAYOUT="${TRAIN_FORWARD_LAYOUT:-sensor_fast}"
export TRAIN_BACKWARD_BACKEND="${TRAIN_BACKWARD_BACKEND:-sensor_tiled}"

# ===============================================================
# settings
# ===============================================================
START_POSE=${START_POSE:-0}
END_POSE=${END_POSE:-9}

# 当前验证配置：单张 NVIDIA A100-SXM4 40GB，物理 GPU 0。
# 可选数量：1 / 2 / 4 / 8
GPU_IDS=(0)

# 自动根据 GPU_IDS 计算使用几张卡
NUM_GPUS=${#GPU_IDS[@]}

# 保留原 Python 文件名，默认使用共享粗模板、批量起点和 Triton 投影。
LOCALIZATION_BACKEND="${LOCALIZATION_BACKEND:-batch}"
case "$LOCALIZATION_BACKEND" in
    batch)
        LOCALIZATION_SCRIPT="single_localization_kde.py"
        PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
        ;;
    *)
        echo "[ERROR] This A100 package uses LOCALIZATION_BACKEND=batch: $LOCALIZATION_BACKEND" >&2
        exit 1
        ;;
esac

if [[ ! "$PROCS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] PROCS_PER_GPU must be a positive integer: $PROCS_PER_GPU" >&2
    exit 1
fi

# 留空时使用 batch Python 入口的默认批大小；这些选项只控制执行分批。
LOCALIZATION_COARSE_BATCH_SIZE="${LOCALIZATION_COARSE_BATCH_SIZE:-}"
LOCALIZATION_SENSOR_BATCH_SIZE="${LOCALIZATION_SENSOR_BATCH_SIZE:-}"
if [ "$LOCALIZATION_BACKEND" = "batch" ]; then
    for batch_size in "$LOCALIZATION_COARSE_BATCH_SIZE" "$LOCALIZATION_SENSOR_BATCH_SIZE"; do
        if [[ -n "$batch_size" && ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
            echo "[ERROR] Localization batch sizes must be positive integers: $batch_size" >&2
            exit 1
        fi
    done
fi

# 坐标统一到哪个 pose 坐标系
TARGET_POSE=${TARGET_POSE:-0}

for pose_value in "$START_POSE" "$END_POSE" "$TARGET_POSE"; do
    if [[ ! "$pose_value" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "[ERROR] Pose IDs must be nonnegative integers without leading zeros" >&2
        exit 1
    fi
done
if [ "$TARGET_POSE" -gt "$START_POSE" ] || [ "$START_POSE" -gt "$END_POSE" ]; then
    echo "[ERROR] Expected TARGET_POSE <= START_POSE <= END_POSE" >&2
    exit 1
fi

GROUP_ID=3
DATA_DIR="data"
SENSOR_LOCATION_FILE="${DATA_DIR}/sensor_location_group03_pose000.txt"

LOSS_THRESHOLD="0.5"

DEVICE_TRAIN=${GPU_IDS[0]}
DEVICE_RECON=${GPU_IDS[0]}

if [ "$NUM_GPUS" -ge 2 ]; then
    DEVICE_REFINE=${GPU_IDS[1]}
else
    DEVICE_REFINE=${GPU_IDS[0]}
fi

LOCALIZATION_NUM_PARTS=$((NUM_GPUS * PROCS_PER_GPU))
if [ "$LOCALIZATION_NUM_PARTS" -gt 256 ]; then
    echo "[ERROR] Localization parts exceed the initial 256 selected sensors" >&2
    exit 1
fi

# 默认固定 run id，避免 checkpoint/log/output 路径随时间变化。
# 如需保留多次运行结果，可在启动前手动指定 RUN_ID。
RUN_ID="${RUN_ID:-stable}"
case "$RUN_ID" in
    *seed*|*Seed*|*SEED*|*1013*)
        RUN_ID="stable"
        ;;
esac
export RUN_ID
LOG_ROOT="logs_group3_pose_range_${RUN_ID}"
RECON_OUTPUT_DIR="recon_outputs_${RUN_ID}"

SCRIPT_START_TS=$(date +%s)

# ===============================================================
# 工具函数
# ===============================================================
format_duration() {
    local total=$1
    printf "%02d:%02d:%02d" $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

print_duration() {
    local name=$1
    local start_ts=$2
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[TIME] ${name} duration: $(format_duration "$elapsed") (${elapsed}s)"
}

wait_many() {
    local status=0

    for item in "$@"; do
        local pid="${item%%:*}"
        local name="${item#*:}"

        wait "$pid"
        local s=$?
        if [ "$s" -ne 0 ]; then
            echo "[ERROR] ${name} failed with exit code ${s}"
            status=1
        else
            echo "[INFO] ${name} finished successfully"
        fi
    done

    return "$status"
}

find_latest_ckpt() {
    local root_dir=$1
    local ckpt_dir
    local ckpt_path

    ckpt_dir=$(find "$root_dir" -maxdepth 1 -type d -name "epoch*" ! -iname "*seed*" ! -name "*1013*" | sort | tail -n 1)

    if [ -z "$ckpt_dir" ]; then
        echo "[ERROR] No checkpoint directory found under ${root_dir}" >&2
        return 1
    fi

    ckpt_path="${ckpt_dir}/model.pt"

    if [ ! -f "$ckpt_path" ]; then
        echo "[ERROR] Checkpoint model not found: ${ckpt_path}" >&2
        return 1
    fi

    echo "$ckpt_path"
}

checkpoint_dir_for_pose() {
    local pose_id=$1
    local ckpt_num=$((pose_id + 1))
    printf "checkpoints_%02d" "$ckpt_num"
}

signal_file_for_pose() {
    local pose_id=$1
    printf "%s/processed_signal_group03_pose%03d.txt" "$DATA_DIR" "$pose_id"
}

simulated_signal_file_for_pose() {
    local pose_id=$1
    printf "simulated_signals_full_4096_group3_pose%d.txt" "$pose_id"
}

refined_location_file_for_pose() {
    local pose_id=$1
    local prev_pose_id=$((pose_id - 1))
    printf "predicted_locations_group3_pose%d_10MHz_baseon%d_corrected_finetuned_masked_10MHz.txt" "$pose_id" "$prev_pose_id"
}

transformed_location_file_for_pose() {
    local pose_id=$1
    printf "transformed_locations_group3_pose%d_to_pose_of_%d.txt" "$pose_id" "$TARGET_POSE"
}

location_file_for_recon_pose() {
    local pose_id=$1

    if [ "$pose_id" -eq "$TARGET_POSE" ]; then
        echo "$SENSOR_LOCATION_FILE"
    elif [ "$pose_id" -eq $((TARGET_POSE + 1)) ]; then
        refined_location_file_for_pose "$pose_id"
    else
        transformed_location_file_for_pose "$pose_id"
    fi
}

extract_last_active_loss() {
    local log_file=$1
    local loss

    loss=$(grep "Active_Loss:" "$log_file" | tail -n 1 | sed -nE 's/.*Active_Loss:[[:space:]]*([^[:space:]|]+).*/\1/p')

    if [ -z "$loss" ]; then
        echo "nan"
    else
        echo "$loss"
    fi
}

loss_greater_than_threshold() {
    local loss=$1
    local threshold=$2

    python - "$loss" "$threshold" <<'PY'
import math
import sys

try:
    loss = float(sys.argv[1])
    threshold = float(sys.argv[2])
except (ValueError, IndexError):
    sys.exit(2)

if not math.isfinite(loss) or not math.isfinite(threshold):
    sys.exit(2)

sys.exit(0 if loss > threshold else 1)
PY
}

# ===============================================================
# 训练 / forward
# ===============================================================
run_train_one_pose() {
    local pose_id=$1
    local device_id=$2
    local ckpt_dir=$3
    local log_dir=$4
    local signal_file
    local start_ts

    signal_file=$(signal_file_for_pose "$pose_id")
    start_ts=$(date +%s)

    mkdir -p "$ckpt_dir"
    mkdir -p "$log_dir"

    echo "[INFO] Starting pose${pose_id} training on GPU${device_id} at $(date)"

    python -u differentiable_radiation_embedding.py \
        --device-id "$device_id" \
        --sensor-location "$SENSOR_LOCATION_FILE" \
        --signal "$signal_file" \
        --save-dir "$ckpt_dir" \
        > "${log_dir}/train_group3_pose${pose_id}_gpu${device_id}.log" 2>&1

    local status=$?
    print_duration "pose${pose_id} training" "$start_ts"
    return "$status"
}

run_forward() {
    local pose_id=$1
    local device_id=$2
    local ckpt_dir=$3
    local log_dir=$4
    local sim_signal_file
    local start_ts

    sim_signal_file=$(simulated_signal_file_for_pose "$pose_id")
    start_ts=$(date +%s)

    echo "[INFO] Starting pose${pose_id} forward on GPU${device_id} at $(date)"

    python -u forward_propagation.py \
        --device-id "$device_id" \
        --sensor-location "$SENSOR_LOCATION_FILE" \
        --ckpt-dir "$ckpt_dir" \
        --ckpt-index -1 \
        --output "$sim_signal_file" \
        > "${log_dir}/forward_group3_pose${pose_id}_gpu${device_id}.log" 2>&1

    local status=$?
    print_duration "pose${pose_id} forward" "$start_ts"
    return "$status"
}

# ===============================================================
# localization
# ===============================================================
run_localization() {
    local pose_id=$1
    local sim_signal_file=$2
    local loc_ckpt_root=$3
    local log_dir=$4
    local out_dir=$5
    local mode=$6
    local start_ts
    local localization_args=()

    if [ "$LOCALIZATION_BACKEND" = "batch" ]; then
        if [ -n "$LOCALIZATION_COARSE_BATCH_SIZE" ]; then
            localization_args+=(--coarse_batch_size "$LOCALIZATION_COARSE_BATCH_SIZE")
        fi
        if [ -n "$LOCALIZATION_SENSOR_BATCH_SIZE" ]; then
            localization_args+=(--sensor_batch_size "$LOCALIZATION_SENSOR_BATCH_SIZE")
        fi
    fi

    start_ts=$(date +%s)

    mkdir -p "$log_dir"
    mkdir -p "$out_dir"

    local loc_ckpt_path
    loc_ckpt_path=$(find_latest_ckpt "$loc_ckpt_root") || return 1

    echo "[INFO] Pose${pose_id} localization checkpoint: ${loc_ckpt_path}"
    echo "[INFO] Starting pose${pose_id} localization mode=${mode} backend=${LOCALIZATION_BACKEND} at $(date)"
    echo "[INFO] NUM_GPUS=${NUM_GPUS}, GPU_IDS=${GPU_IDS[*]}, PROCS_PER_GPU=${PROCS_PER_GPU}, PARTS=${LOCALIZATION_NUM_PARTS}"

    rm -f "${out_dir}/pred_pose${pose_id}_part"*.txt

    local loc_items=()

    for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
        local gpu_id=${GPU_IDS[$gpu_idx]}

        for local_part in $(seq 0 $((PROCS_PER_GPU - 1))); do
            part_id=$((gpu_idx * PROCS_PER_GPU + local_part))

            if [ "$mode" = "256" ]; then
                start_id=$((part_id * 4))
                step=$((LOCALIZATION_NUM_PARTS * 4))
                sensor_ids=$(seq -s, "$start_id" "$step" 1023)
            elif [ "$mode" = "1024" ]; then
                start_id=$part_id
                step=$LOCALIZATION_NUM_PARTS
                sensor_ids=$(seq -s, "$start_id" "$step" 1023)
            else
                echo "[ERROR] Unknown localization mode: $mode"
                return 1
            fi

            CUDA_VISIBLE_DEVICES="$gpu_id" python -u "$LOCALIZATION_SCRIPT" \
                --signal "$sim_signal_file" \
                --sensor_gt "$SENSOR_LOCATION_FILE" \
                --ckpt "$loc_ckpt_path" \
                --out "${out_dir}/pred_pose${pose_id}_part${part_id}.txt" \
                --sensor_ids "$sensor_ids" \
                --gpu 0 \
                --use_kde \
                "${localization_args[@]}" \
                > "${log_dir}/localization_pose${pose_id}_part${part_id}_gpu${gpu_id}_${mode}.log" 2>&1 &

            local pid=$!
            loc_items+=("${pid}:localization pose${pose_id} part${part_id} gpu${gpu_id} mode${mode}")
        done
    done

    wait_many "${loc_items[@]}"
    local status=$?

    print_duration "pose${pose_id} localization ${mode}" "$start_ts"
    return "$status"
}

# ===============================================================
# correction / refine / sequential / recon
# ===============================================================
run_position_refine() {
    local pose_id=$1
    local ckpt_dir=$2
    local log_dir=$3
    local tag=$4
    local start_ts
    local refine_log

    start_ts=$(date +%s)
    refine_log="${log_dir}/position_correction_and_refine_pose${pose_id}_${tag}.log"

    echo "[INFO] Starting pose${pose_id} position correction/refine tag=${tag} at $(date)"

    python -u position_correction_and_refine.py \
        --pose-id "$pose_id" \
        --ckpt-dir "$ckpt_dir" \
        --device-id "$DEVICE_REFINE" \
        --sensor-location "$SENSOR_LOCATION_FILE" \
        --num-localization-parts "$LOCALIZATION_NUM_PARTS" \
        > "$refine_log" 2>&1

    local status=$?
    print_duration "pose${pose_id} position correction/refine ${tag}" "$start_ts"

    if [ "$status" -ne 0 ]; then
        echo "[ERROR] pose${pose_id} position correction/refine ${tag} failed"
        return "$status"
    fi

    local loss
    loss=$(extract_last_active_loss "$refine_log")
    echo "[INFO] pose${pose_id} ${tag} last Active_Loss=${loss}"
    echo "$loss" > "${log_dir}/pose${pose_id}_${tag}_last_active_loss.txt"

    return 0
}

run_sequential_correction_if_needed() {
    local pose_id=$1
    local log_dir=$2

    if [ "$pose_id" -le $((TARGET_POSE + 1)) ]; then
        echo "[INFO] Pose${pose_id} does not need sequential correction"
        return 0
    fi

    local start_ts
    start_ts=$(date +%s)

    echo "[INFO] Starting pose${pose_id} sequential correction to pose${TARGET_POSE} frame at $(date)"

    python -u sequential_correction.py \
        --pose-id "$pose_id" \
        --target-pose-id "$TARGET_POSE" \
        --group-id "$GROUP_ID" \
        --freq-tag 10MHz \
        --ref-file "$SENSOR_LOCATION_FILE" \
        > "${log_dir}/sequential_correction_pose${pose_id}.log" 2>&1

    local status=$?
    print_duration "pose${pose_id} sequential correction" "$start_ts"
    return "$status"
}

run_joint_recon_until_pose() {
    local max_pose=$1
    local log_dir=$2
    local start_ts

    start_ts=$(date +%s)

    local signal_files=()
    local location_files=()

    for pose_id in $(seq "$TARGET_POSE" "$max_pose"); do
        signal_files+=("$(signal_file_for_pose "$pose_id")")
        location_files+=("$(location_file_for_recon_pose "$pose_id")")
    done

    echo "[INFO] Starting joint reconstruction pose${TARGET_POSE}-to-pose${max_pose} at $(date)"

    python -u joint_recon.py \
        --device-id "$DEVICE_RECON" \
        --method DAS \
        --output-dir "$RECON_OUTPUT_DIR" \
        --output-prefix "3Dpano_group3_pose${TARGET_POSE}_to_${max_pose}_loc_pred" \
        --signal-files "${signal_files[@]}" \
        --location-files "${location_files[@]}" \
        > "${log_dir}/joint_recon_pose${TARGET_POSE}_to_pose${max_pose}.log" 2>&1

    local status=$?
    print_duration "joint reconstruction pose${TARGET_POSE}-to-pose${max_pose}" "$start_ts"
    return "$status"
}

# ===============================================================
# 检查输入
# ===============================================================
check_required_files() {
    if [ "$NUM_GPUS" -ne 1 ] && [ "$NUM_GPUS" -ne 2 ] && [ "$NUM_GPUS" -ne 4 ] && [ "$NUM_GPUS" -ne 8 ]; then
        echo "[ERROR] Number of GPU_IDS must be one of: 1, 2, 4, 8"
        exit 1
    fi

    if [ ! -f "$LOCALIZATION_SCRIPT" ]; then
        echo "[ERROR] Missing localization backend script: $LOCALIZATION_SCRIPT"
        exit 1
    fi

    local required_file
    for required_file in differentiable_radiation_embedding.py forward_propagation.py \
        position_correction_and_refine.py sequential_correction.py joint_recon.py \
        lib/forward.py lib/forward_batch.py lib/forward_triton.py lib/runtime_config.py \
        lib/time_scatter.py lib/training_backward_tiled.py lib/training_forward_looped.py \
        lib/training_backward_grouped.py lib/refine_direct.py lib/localization_graph.py \
        lib/training_forward_shared.py lib/training_forward_shared.cu lib/nvrtc_driver.py; do
        if [ ! -f "$required_file" ]; then
            echo "[ERROR] Missing pipeline file: $required_file" >&2
            exit 1
        fi
    done

    if [ ! -f "$SENSOR_LOCATION_FILE" ]; then
        echo "[ERROR] Missing sensor location file: $SENSOR_LOCATION_FILE"
        exit 1
    fi

    for pose_id in $(seq "$TARGET_POSE" "$END_POSE"); do
        local signal_file
        signal_file=$(signal_file_for_pose "$pose_id")
        if [ ! -f "$signal_file" ]; then
            echo "[ERROR] Missing signal file: $signal_file"
            exit 1
        fi
    done

    if [ "$START_POSE" -gt "$TARGET_POSE" ]; then
        local previous_ckpt_root
        previous_ckpt_root=$(checkpoint_dir_for_pose $((START_POSE - 1)))
        if [ ! -d "$previous_ckpt_root" ] || ! find_latest_ckpt "$previous_ckpt_root" >/dev/null; then
            echo "[ERROR] Resume requires a trained previous pose in $previous_ckpt_root" >&2
            exit 1
        fi
        for pose_id in $(seq "$TARGET_POSE" $((START_POSE - 1))); do
            local loc_file
            loc_file=$(location_file_for_recon_pose "$pose_id")
            if [ ! -f "$loc_file" ]; then
                echo "[ERROR] Missing previous location file for pose${pose_id}: $loc_file"
                exit 1
            fi
        done
    fi
}

# ===============================================================
# pose 后半段：localization/refine/sequential/recon
# ===============================================================
run_pose_after_forward_adaptive() {
    local pose_id=$1
    local ckpt_dir=$2
    local prev_ckpt_dir=$3
    local log_dir=$4
    local out_dir=$5
    local sim_signal_file
    local loss256
    local loss1024

    sim_signal_file=$(simulated_signal_file_for_pose "$pose_id")

    run_localization "$pose_id" "$sim_signal_file" "$prev_ckpt_dir" "$log_dir" "$out_dir" "256" || return 1
    run_position_refine "$pose_id" "$prev_ckpt_dir" "$log_dir" "256" || return 1

    loss256=$(cat "${log_dir}/pose${pose_id}_256_last_active_loss.txt")

    loss_greater_than_threshold "$loss256" "$LOSS_THRESHOLD"
    local cmp_status=$?

    if [ "$cmp_status" -eq 0 ]; then
        echo "[WARN] pose${pose_id} 256-sensor refine failed threshold: loss=${loss256} > ${LOSS_THRESHOLD}"
        echo "[WARN] Switching pose${pose_id} to 1024-sensor localization and overwriting pose${pose_id} results"

        run_localization "$pose_id" "$sim_signal_file" "$prev_ckpt_dir" "$log_dir" "$out_dir" "1024" || return 1
        run_position_refine "$pose_id" "$prev_ckpt_dir" "$log_dir" "1024" || return 1

        loss1024=$(cat "${log_dir}/pose${pose_id}_1024_last_active_loss.txt")

        loss_greater_than_threshold "$loss1024" "$LOSS_THRESHOLD"
        local cmp1024_status=$?

        if [ "$cmp1024_status" -eq 0 ]; then
            echo "[ERROR] pose${pose_id} 1024-sensor refine still failed threshold: loss=${loss1024} > ${LOSS_THRESHOLD}"
            return 1
        elif [ "$cmp1024_status" -eq 1 ]; then
            echo "[INFO] pose${pose_id} 1024-sensor refine passed: loss=${loss1024} <= ${LOSS_THRESHOLD}"
        else
            echo "[ERROR] pose${pose_id} 1024-sensor refine loss is non-finite, unparsable, or could not be checked"
            return 1
        fi
    elif [ "$cmp_status" -eq 1 ]; then
        echo "[INFO] pose${pose_id} 256-sensor refine passed: loss=${loss256} <= ${LOSS_THRESHOLD}"
    else
        echo "[ERROR] pose${pose_id} 256-sensor refine loss is non-finite, unparsable, or could not be checked"
        return 1
    fi

    run_sequential_correction_if_needed "$pose_id" "$log_dir" || return 1
    run_joint_recon_until_pose "$pose_id" "$log_dir" || return 1

    return 0
}

# ===============================================================
# 单帧完整自适应流程
# ===============================================================
run_one_pose_adaptive() {
    local pose_id=$1
    local pose_start_ts
    local ckpt_dir
    local prev_pose_id
    local prev_ckpt_dir
    local log_dir
    local out_dir

    pose_start_ts=$(date +%s)
    ckpt_dir=$(checkpoint_dir_for_pose "$pose_id")
    prev_pose_id=$((pose_id - 1))
    prev_ckpt_dir=$(checkpoint_dir_for_pose "$prev_pose_id")
    log_dir="${LOG_ROOT}/pose${pose_id}"
    out_dir="outputs_${pose_id}"

    mkdir -p "$log_dir"
    mkdir -p "$out_dir"
    mkdir -p "$ckpt_dir"

    echo "############################################################"
    echo "[INFO] Starting adaptive pose${pose_id} pipeline at $(date)"
    echo "[INFO] ckpt_dir=${ckpt_dir}"
    echo "[INFO] prev_ckpt_dir=${prev_ckpt_dir}"
    echo "[INFO] log_dir=${log_dir}"
    echo "[INFO] out_dir=${out_dir}"
    echo "############################################################"

    run_train_one_pose "$pose_id" "$DEVICE_TRAIN" "$ckpt_dir" "$log_dir" || return 1
    run_forward "$pose_id" "$DEVICE_TRAIN" "$ckpt_dir" "$log_dir" || return 1
    run_pose_after_forward_adaptive "$pose_id" "$ckpt_dir" "$prev_ckpt_dir" "$log_dir" "$out_dir" || return 1

    print_duration "pose${pose_id} adaptive pipeline" "$pose_start_ts"
    echo "[SUCCESS] pose${pose_id} adaptive pipeline completed at $(date)"
    return 0
}

# ===============================================================
# 多卡初始化：pose0/pose1 并行
# ===============================================================
run_parallel_initial_pose0_pose1() {
    local start_ts
    start_ts=$(date +%s)

    local pose0=$TARGET_POSE
    local pose1=$((TARGET_POSE + 1))

    local ckpt0
    local ckpt1
    local log0
    local log1
    local out1

    ckpt0=$(checkpoint_dir_for_pose "$pose0")
    ckpt1=$(checkpoint_dir_for_pose "$pose1")
    log0="${LOG_ROOT}/pose${pose0}"
    log1="${LOG_ROOT}/pose${pose1}"
    out1="outputs_${pose1}"

    mkdir -p "$ckpt0" "$ckpt1" "$log0" "$log1" "$out1"

    echo "[INFO] Multi-GPU initial stage: pose${pose0}/pose${pose1} train in parallel"

    run_train_one_pose "$pose0" "${GPU_IDS[0]}" "$ckpt0" "$log0" &
    local train_pid0=$!

    run_train_one_pose "$pose1" "${GPU_IDS[1]}" "$ckpt1" "$log1" &
    local train_pid1=$!

    wait_many \
        "$train_pid0:train pose${pose0} gpu${GPU_IDS[0]}" \
        "$train_pid1:train pose${pose1} gpu${GPU_IDS[1]}" || return 1

    echo "[INFO] Multi-GPU initial stage: pose${pose0}/pose${pose1} forward in parallel"

    run_forward "$pose0" "${GPU_IDS[0]}" "$ckpt0" "$log0" &
    local forward_pid0=$!

    run_forward "$pose1" "${GPU_IDS[1]}" "$ckpt1" "$log1" &
    local forward_pid1=$!

    wait_many \
        "$forward_pid0:forward pose${pose0} gpu${GPU_IDS[0]}" \
        "$forward_pid1:forward pose${pose1} gpu${GPU_IDS[1]}" || return 1

    run_pose_after_forward_adaptive "$pose1" "$ckpt1" "$ckpt0" "$log1" "$out1" || return 1

    print_duration "parallel initial pose${pose0}/pose${pose1} stage" "$start_ts"
    return 0
}

# ===============================================================
# 主流程
# ===============================================================
echo "============================================================"
echo "[INFO] Pose range pipeline started at $(date)"
echo "[INFO] START_POSE=${START_POSE}"
echo "[INFO] END_POSE=${END_POSE}"
echo "[INFO] TARGET_POSE=${TARGET_POSE}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}"
echo "[INFO] GPU_IDS=${GPU_IDS[*]}"
echo "[INFO] LOCALIZATION_BACKEND=${LOCALIZATION_BACKEND}"
echo "[INFO] LOCALIZATION_SCRIPT=${LOCALIZATION_SCRIPT}"
echo "[INFO] REFINE_BACKEND=${REFINE_BACKEND}"
echo "[INFO] LOCALIZATION_FINE_EXECUTION=${LOCALIZATION_FINE_EXECUTION}"
echo "[INFO] LOCALIZATION_PROJECTOR=${LOCALIZATION_PROJECTOR}"
echo "[INFO] TRAIN_TIME_SCATTER=${TRAIN_TIME_SCATTER}"
echo "[INFO] TRAIN_FORWARD_PROJECTOR=${TRAIN_FORWARD_PROJECTOR}"
echo "[INFO] TRAIN_FORWARD_LAYOUT=${TRAIN_FORWARD_LAYOUT}"
echo "[INFO] TRAIN_BACKWARD_BACKEND=${TRAIN_BACKWARD_BACKEND}"
echo "[INFO] PROCS_PER_GPU=${PROCS_PER_GPU}"
if [ "$LOCALIZATION_BACKEND" = "batch" ]; then
    echo "[INFO] LOCALIZATION_COARSE_BATCH_SIZE=${LOCALIZATION_COARSE_BATCH_SIZE:-python-default}"
    echo "[INFO] LOCALIZATION_SENSOR_BATCH_SIZE=${LOCALIZATION_SENSOR_BATCH_SIZE:-python-default}"
fi
echo "[INFO] LOCALIZATION_NUM_PARTS=${LOCALIZATION_NUM_PARTS}"
echo "[INFO] LOSS_THRESHOLD=${LOSS_THRESHOLD}"
echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo "[INFO] REPRO_DETERMINISTIC=${REPRO_DETERMINISTIC}"
echo "[INFO] REPRO_DETERMINISTIC_WARN_ONLY=${REPRO_DETERMINISTIC_WARN_ONLY}"
echo "[INFO] REPRO_ALLOW_TF32=${REPRO_ALLOW_TF32}"
echo "[INFO] REPRO_STRICT_ATOMICS=${REPRO_STRICT_ATOMICS}"
echo "[INFO] REPRO_FIXED_POINT_SCALE=${REPRO_FIXED_POINT_SCALE}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] RECON_OUTPUT_DIR=${RECON_OUTPUT_DIR}"
echo "[INFO] SENSOR_LOCATION_FILE=${SENSOR_LOCATION_FILE}"
echo "============================================================"

check_required_files

if [ "$CHECK_ONLY" -eq 1 ]; then
    python - "${GPU_IDS[@]}" <<'PY'
import sys
import numpy
import scipy
import torch
import triton
from lib import forward_batch, forward_triton, time_scatter, training_backward_tiled, training_forward_looped, training_backward_grouped, refine_direct, localization_graph, training_forward_shared, nvrtc_driver

if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is unavailable in the selected Python environment")
for device_id in map(int, sys.argv[1:]):
    if not 0 <= device_id < torch.cuda.device_count():
        raise SystemExit(f"[ERROR] GPU {device_id} is not visible")
    print(f"[CHECK] GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
print(f"[CHECK] Python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"CUDA {torch.version.cuda}, Triton {triton.__version__}, "
      f"NumPy {numpy.__version__}, SciPy {scipy.__version__}")
print("[CHECK] Pipeline files, input files, imports and GPU checks passed. No reconstruction started.")
PY
    exit "$?"
fi

mkdir -p "$LOG_ROOT" "$RECON_OUTPUT_DIR"

if [ "$START_POSE" -eq "$TARGET_POSE" ]; then
    mkdir -p "${LOG_ROOT}/pose${TARGET_POSE}"

    run_joint_recon_until_pose "$TARGET_POSE" "${LOG_ROOT}/pose${TARGET_POSE}" || exit 1

    if [ "$NUM_GPUS" -ge 2 ] && [ "$END_POSE" -ge $((TARGET_POSE + 1)) ]; then
        run_parallel_initial_pose0_pose1 || exit 1
        first_dynamic_pose=$((TARGET_POSE + 2))
    else
        target_ckpt_dir=$(checkpoint_dir_for_pose "$TARGET_POSE")
        run_train_one_pose "$TARGET_POSE" "${GPU_IDS[0]}" "$target_ckpt_dir" "${LOG_ROOT}/pose${TARGET_POSE}" || exit 1
        run_forward "$TARGET_POSE" "${GPU_IDS[0]}" "$target_ckpt_dir" "${LOG_ROOT}/pose${TARGET_POSE}" || exit 1
        first_dynamic_pose=$((TARGET_POSE + 1))
    fi
else
    first_dynamic_pose="$START_POSE"
fi

if [ "$first_dynamic_pose" -le "$END_POSE" ]; then
    for POSE_ID in $(seq "$first_dynamic_pose" "$END_POSE"); do
        run_one_pose_adaptive "$POSE_ID"
        status=$?

        if [ "$status" -ne 0 ]; then
            echo "[ERROR] pose${POSE_ID} adaptive pipeline failed. Stop remaining poses."
            print_duration "total pose range pipeline" "$SCRIPT_START_TS"
            exit 1
        fi
    done
fi

print_duration "total pose range pipeline" "$SCRIPT_START_TS"
echo "[SUCCESS] Pose range pipeline completed at $(date)"
