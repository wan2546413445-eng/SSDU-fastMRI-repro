#!/usr/bin/env bash
set -euo pipefail

KSPACE_DIR="/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/kspace"
MAPS_DIR="/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/maps"
OUT_ROOT="outputs/zs_ssl_vol0_nocrop"
LOG_DIR="logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

for s in $(seq 0 35); do
  MAT_PATH="$OUT_ROOT/data_vol0_slice${s}_nocrop.mat"
  RUN_DIR="$OUT_ROOT/vol0_slice${s}_nocrop"
  LOG_PATH="$LOG_DIR/zs_ssl_vol0_slice${s}.log"

  python scripts/convert_hfs_knee_slice_to_zs_ssl_mat.py \
    --kspace_dir "$KSPACE_DIR" \
    --maps_dir "$MAPS_DIR" \
    --volume_index 0 \
    --slice_index "$s" \
    --acc 4 \
    --acs 24 \
    --crop_size 0 \
    --out "$MAT_PATH"

  python scripts/train_zs_ssl_hfs_knee.py \
    --data_path "$MAT_PATH" \
    --output_dir "$RUN_DIR" \
    --epochs 50 \
    --stop_training 10 \
    --num_reps 2 \
    --nb_unroll_blocks 2 \
    --nb_res_blocks 2 \
    --cg_iter 2 \
    --seed 2026 \
    --acs_block 24 2>&1 | tee "$LOG_PATH"
done
