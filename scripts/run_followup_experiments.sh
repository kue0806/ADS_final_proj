#!/usr/bin/env bash
# Follow-up experiments to harden findings F1, F2, F5.
# Run inside the WSL terminal (NOT Git-Bash):  bash scripts/run_followup_experiments.sh [exp]
#
#   exp = 2   F5 artifact test on existing bilinear activations (no extract)   ~5 min  ★★★
#   exp = 3   F2 refinement: single-fault-only alignment (no extract)          ~5 min  ★★★
#   exp = 1   F1 natural-image (CIFAR-10) baseline: extract+SAE+MS             ~20 min ★★
#   exp = 2b  F5 preprocessing: nearest-upsample extract+SAE+artifact          ~20 min ★
#   exp = all run 2, 3, 1, 2b in that order
#
# Default (no arg) runs the two highest-value, no-extract experiments: 2 and 3.

set -euo pipefail
cd "$(dirname "$0")/.."   # project root

PY=python
DINO=facebook/dinov2-base
ACTS=data/activations/dinov2-base__L-2__train
SAE=outputs/sae_ckpts/sae__dinov2-base__L-2__train__d12288__k32__s30000.pt
RAW=data/raw/MixedWM38.npz

banner() { echo; echo "========== $* =========="; echo; }

exp2() {  # F5 artifact (bilinear, existing activations)
  banner "EXP 2 — F5 artifact test (bilinear, existing activations)"
  $PY scripts/measure_artifact.py --acts_dir "$ACTS" --sae "$SAE" --raw "$RAW"
}

exp3() {  # F2 refinement (existing activations)
  # NOTE: the patch<->native mapping was corrected (faithful bilinear instead
  # of zero-padded block pooling), so we RE-RUN the full E2 too to refresh the
  # baseline correlation, then the single-fault-only version.
  banner "EXP 3a — E2 full re-run (corrected mapping)"
  $PY scripts/measure_alignment.py --acts_dir "$ACTS" --sae "$SAE" --raw "$RAW" \
      --chunk_images 128
  banner "EXP 3b — F2 refinement: single-fault-only alignment"
  $PY scripts/measure_alignment.py --acts_dir "$ACTS" --sae "$SAE" --raw "$RAW" \
      --chunk_images 128 --single_fault_only
}

exp1() {  # F1 natural-image baseline: CIFAR-10
  banner "EXP 1 — F1 natural-image (CIFAR-10) baseline"
  local NAT=data/activations/NAT-cifar10-dinov2-base__L-2__train__N8000
  local NSAE=outputs/sae_ckpts/sae__NAT-cifar10-dinov2-base__L-2__train__N8000__d12288__k32__s30000.pt
  $PY scripts/extract_natural_baseline.py --backbone "$DINO" --n 8000 --batch 64
  $PY scripts/train_sae.py --acts_dir "$NAT" \
      --steps 30000 --batch 1024 --k 32 --dict_mult 16 \
      --chunk_images 2000 --log_every 5000
  $PY scripts/measure_monosemanticity.py --acts_dir "$NAT" --sae "$NSAE" \
      --chunk_images 512
}

exp2b() {  # F5 preprocessing: nearest upsample (separate dir, preserves bilinear)
  banner "EXP 2b — F5 preprocessing: nearest-upsample"
  local NROOT=data/activations/nearest
  local NACTS="$NROOT/dinov2-base__L-2__train"
  local NSAE=outputs/sae_ckpts/sae__dinov2-base__L-2__train__d12288__k32__s30000__nearest.pt
  $PY scripts/extract_activations.py --backbone "$DINO" --split train --batch 64 \
      --upsample nearest --out_root "$NROOT"
  # train_sae names ckpt from acts_dir basename, so it would collide with the
  # bilinear ckpt; train into a temp out_dir then move.
  $PY scripts/train_sae.py --acts_dir "$NACTS" \
      --steps 30000 --batch 1024 --k 32 --dict_mult 16 \
      --chunk_images 2000 --log_every 5000 \
      --out_dir outputs/sae_ckpts/nearest
  $PY scripts/measure_artifact.py --acts_dir "$NACTS" \
      --sae outputs/sae_ckpts/nearest/sae__dinov2-base__L-2__train__d12288__k32__s30000.pt \
      --raw "$RAW"
}

case "${1:-default}" in
  2)       exp2 ;;
  3)       exp3 ;;
  1)       exp1 ;;
  2b)      exp2b ;;
  all)     exp2; exp3; exp1; exp2b ;;
  default) exp2; exp3 ;;
  *) echo "usage: bash scripts/run_followup_experiments.sh [2|3|1|2b|all|default]"; exit 1 ;;
esac

banner "DONE"
