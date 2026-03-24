#!/usr/bin/env bash

set -euo pipefail

# Defaults (override with env vars if needed)
DATA_PATH="${DATA_PATH:-/home/berkay/Desktop/research/datasets/NeoRadiumTDLdataset/test/TDLA}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SNR="${SNR:-20}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
DEVICE="${DEVICE:-cuda:0}"

# CeViT model defaults from train/evaluate scripts
TOKEN_EMB_DIM="${TOKEN_EMB_DIM:-168}"
PATCH_DIM="${PATCH_DIM:-40}"
MODEL_DIM="${MODEL_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.0}"
PATCH_H="${PATCH_H:-10}"
PATCH_W="${PATCH_W:-4}"
ACTIVATION="${ACTIVATION:-gelu}"
NUM_SUBCARRIERS="${NUM_SUBCARRIERS:-120}"
NUM_SYMBOLS="${NUM_SYMBOLS:-14}"

echo "Running CeViT benchmark on DATA_PATH=${DATA_PATH}"

cmd=(
  python3 benchmark_cevit.py
  --data_path "${DATA_PATH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --snr "${SNR}"
  --warmup_batches "${WARMUP_BATCHES}"
  --device "${DEVICE}"
  --token_emb_dim "${TOKEN_EMB_DIM}"
  --patch_dim "${PATCH_DIM}"
  --model_dim "${MODEL_DIM}"
  --num_heads "${NUM_HEADS}"
  --dropout "${DROPOUT}"
  --patch_h "${PATCH_H}"
  --patch_w "${PATCH_W}"
  --activation "${ACTIVATION}"
  --num_subcarriers "${NUM_SUBCARRIERS}"
  --num_symbols "${NUM_SYMBOLS}"
)

"${cmd[@]}"
