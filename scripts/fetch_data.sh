#!/usr/bin/env bash
# Fetch the raw dataset (516 MB). Not committed - see .gitignore.
#
# Source: "Customer Support on Twitter" (Kaggle: thoughtvector/customer-support-on-twitter).
# Pulled from a HuggingFace mirror of the identical twcs.csv so the pipeline
# runs without Kaggle credentials.
set -euo pipefail
mkdir -p "$(dirname "$0")/../data"
OUT="$(dirname "$0")/../data/twcs.csv"
if [ -f "$OUT" ]; then
  echo "already present: $OUT"; exit 0
fi
curl -L --fail -o "$OUT" \
  "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
echo "downloaded: $OUT ($(stat -c%s "$OUT") bytes; expected 516508641)"
