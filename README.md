# Sparse Autoencoder Analysis of DINOv2 Representations on Wafer Bin Maps

Interpretability study of a frozen DINOv2 backbone on wafer bin maps (WBM).
A top-k sparse autoencoder (SAE) decomposes the frozen patch features into an
interpretable dictionary, which is then analyzed for class vs. spatial
selectivity and compared across backbones (self-supervised / supervised /
random).
 
## Data

- **MixedWM38** (38,015 wafer bin maps, 52×52): place `MixedWM38.npz` in
  `data/raw/`. Source: https://github.com/Junliangwangdhu/WaferMap
- **CIFAR-10** (natural-image reference): downloaded automatically by
  `extract_natural_baseline.py`.

## Repository layout

```
src/
  data/wbm_loader.py        # MixedWM38 loading & preprocessing
  models/dino_extractor.py  # frozen ViT feature extractor (supports 'random:<id>')
  sae/topk_sae.py           # top-k sparse autoencoder
  eval/                     # monosemanticity, alignment, artifact, classify metrics
scripts/                    # runnable pipeline (below)
configs/default.yaml        # default hyperparameters
```

## Reproducing the main results

Run from the repository root. Default backbone is `facebook/dinov2-base`;
use `--backbone facebook/vit-base-patch16-224` or `--backbone random:facebook/vit-base-patch16-224`
for the comparison backbones.

```bash
# 1. Extract frozen patch features for the WBM train split
python scripts/extract_activations.py --backbone facebook/dinov2-base --split train

# 2. Train the top-k SAE (dict = 16x feature dim, k = 32)
python scripts/train_sae.py \
  --acts_dir data/activations/dinov2-base__L-2__train \
  --steps 30000 --k 32 --dict_mult 16

# 3a. Monosemanticity (class / spatial selectivity)   [Result 1]
python scripts/measure_monosemanticity.py \
  --acts_dir data/activations/dinov2-base__L-2__train \
  --sae outputs/sae_ckpts/sae__dinov2-base__L-2__train__d12288__k32__s30000.pt
#   add --balance_classes for the class-balanced control

# 3b. Linear-probe accuracy: raw vs. SAE features      [Result 2]
python scripts/run_classifier.py \
  --acts_dir data/activations/dinov2-base__L-2__train \
  --sae outputs/sae_ckpts/sae__dinov2-base__L-2__train__d12288__k32__s30000.pt
```

For the natural-image reference, replace step 1 with
`python scripts/extract_natural_baseline.py` and repeat steps 2–3a.
Additional analyses (defect-mask alignment, signal-vs-artifact) are in
`scripts/measure_alignment.py` and `scripts/measure_artifact.py`;
`scripts/run_followup_experiments.sh` bundles the control experiments.
Result figures are produced by the `scripts/make_*.py` scripts.

## Notes

- The backbone is always frozen; only the SAE (and a linear probe) is trained.
- Large artifacts (`data/`, `outputs/`) are git-ignored.
- Experiments were run on a single NVIDIA RTX 3090 (CUDA 12.1).
