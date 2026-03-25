#!/usr/bin/env bash
# Continue training tdla_2 … tdle_2 for exp1–exp5: load prior best.pt (see configs/cont_train_tdlx_2.yaml),
# same out_dir and TensorBoard log dir as phase 1 (runs/expY/tdlx_2/tb).
# Skips exp1 + tdla_2 (we ran that combination manually).
# Uses the same --seed as train_many.sh (expN -> seed N) so train/val split matches phase 1.
#
# Usage: ./continue_train_many.sh [device]
#   device: optional first argument, or set DEVICE env (default cuda:0).
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  DEVICE="$1"
  shift
fi
DEVICE="${DEVICE:-cuda:0}"

RUNS=(tdla_2 tdlb_2 tdlc_2 tdld_2 tdle_2)

for EXP in exp1 exp2 exp3 exp4 exp5; do
  SEED="${EXP#exp}"
  for RUN in "${RUNS[@]}"; do
    CFG="configs/cont_train_${RUN}.yaml"
    RESUME="runs/${EXP}/${RUN}/best.pt"
    OUT_DIR="runs/${EXP}/${RUN}"

    if [[ "${EXP}" == "exp1" && "${RUN}" == "tdla_2" ]]; then
      echo "skip: exp1 tdla_2 (manual experiment)" >&2
      continue
    fi

    if [[ ! -f "${RESUME}" ]]; then
      echo "skip: missing checkpoint ${RESUME}" >&2
      continue
    fi

    echo "=== continue | ${EXP} | device=${DEVICE} | seed=${SEED} | ${RUN} | resume=${RESUME} ==="
    python3 train.py "${CFG}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --out_dir "${OUT_DIR}" \
      --resume "${RESUME}"
  done
done
