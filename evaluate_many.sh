#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
EXP="${1:-exp1}"

DATA_ROOT="/opt/shared/datasets/NeoRadiumTDLdataset/test"

RUNS=(
  "tdla_2"
  "tdlb_2"
  "tdlc_2"
  "tdld_2"
  "tdle_2"

  "tdla_23"
  "tdlb_23"
  "tdlc_23"
  "tdld_23"
  "tdle_23"

  "tdla_2711"
  "tdlb_2711"
  "tdlc_2711"
  "tdld_2711"
  "tdle_2711"
)

for RUN in "${RUNS[@]}"; do
  # tdla_2 -> TDLA, tdle_2711 -> TDLE
  TDL_VARIANT="${RUN%%_*}"           # tdla
  TDL_DIR="${TDL_VARIANT^^}"        # TDLA  (bash uppercase)

  # _2 -> "2", _23 -> "2 3", _2711 -> "2 7 11"
  PILOT_SUFFIX="${RUN#*_}"           # 2 / 23 / 2711
  case "${PILOT_SUFFIX}" in
    2)    PILOT_SYMBOLS="2"       ;;
    23)   PILOT_SYMBOLS="2 3"     ;;
    2711) PILOT_SYMBOLS="2 7 11"  ;;
    *)    echo "Unknown pilot suffix '${PILOT_SUFFIX}' for run '${RUN}'" >&2; exit 1 ;;
  esac

  CHECKPOINT="runs/${EXP}/${RUN}/best.pt"
  DATA_PATH="${DATA_ROOT}/${TDL_DIR}/"

  echo "=== ${EXP} | ${RUN} | pilots=${PILOT_SYMBOLS} ==="
  # shellcheck disable=SC2086
  python3 evaluate.py \
    --data_path "${DATA_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --pilot_symbols ${PILOT_SYMBOLS} \
    --device "${DEVICE}"
done
