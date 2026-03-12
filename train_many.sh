#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"

CONFIGS=(
  "configs/train_tdla_2.yaml"
  "configs/train_tdlb_2.yaml"
  "configs/train_tdlc_2.yaml"
  "configs/train_tdld_2.yaml"
  "configs/train_tdle_2.yaml"

  "configs/train_tdla_23.yaml"
  "configs/train_tdlb_23.yaml"
  "configs/train_tdlc_23.yaml"
  "configs/train_tdld_23.yaml"
  "configs/train_tdle_23.yaml"

  "configs/train_tdla_2711.yaml"
  "configs/train_tdlb_2711.yaml"
  "configs/train_tdlc_2711.yaml"
  "configs/train_tdld_2711.yaml"
  "configs/train_tdle_2711.yaml"
)

# Four experiment sets: exp2, exp3, exp4, exp5.
# We map expN -> seed N so each set is a different seed.
for EXP in exp2 exp3 exp4 exp5; do
  SEED="${EXP#exp}" # exp2 -> 2
  for CFG in "${CONFIGS[@]}"; do
    RUN_NAME="$(basename "${CFG}" .yaml)"
    RUN_NAME="${RUN_NAME#train_}"
    OUT_DIR="runs/${EXP}/${RUN_NAME}"

    echo "=== ${EXP} | seed=${SEED} | ${CFG} -> ${OUT_DIR} ==="
    python3 train.py "${CFG}" --device "${DEVICE}" --seed "${SEED}" --out_dir "${OUT_DIR}"
  done
done
