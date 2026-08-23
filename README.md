# VasCA-Net (unofficial PyTorch implementation)

A PyTorch re-implementation of:

> Ma, Z., Li, X., Zhao, Y., & Wang, H. (2026). **VasCA-Net: A vascular channel
> attention network for retinal vessel segmentation.** *Expert Systems With
> Applications, 303*, 130591. https://doi.org/10.1016/j.eswa.2025.130591

This is a from-scratch build based on the architecture and equations
described in the paper (Fig. 2, Fig. 3, Eqs. 1–18). No official code was
released alongside the paper, so some low-level implementation choices
(e.g. exact channel-reduction ratios inside EConv) are reasonable
interpretations of the figures/text rather than verified against the
authors' code — see [Implementation notes](#implementation-notes) below.

## Architecture

```
Input (H×W×3)
 │
 ├─ EConv ─ pool ─ EConv ─ pool ─ EConv ─ pool ─ EConv ─ pool ─ Bottleneck
 │     │              │              │              │
 │   MSCA           MSCA           MSCA           MSCA          (skip attention)
 │     │              │              │              │
 └─ DConv ← DConv ← DConv ← DConv ←──┘
         (decoder upsamples back to H×W, 1×1 conv head → vessel mask)
```

- **EConv block** (`models/econv.py`): multi-branch encoder block
  (1×1 conv, stacked 3×3 convs, and a GAP branch fused by element-wise
  multiplication + concatenation) that mitigates feature degradation during
  downsampling.
- **MSCA** (`models/msca.py`): multi-scale channel attention applied to
  every skip connection, using 1×1/3×3/5×5 convolutions on both
  average- and max-pooled descriptors, per Fig. 3 / Eqs. (1)–(6).
- **DConv block** (`models/dconv.py`): decoder block that upsamples the
  low-level feature map, concatenates with the (MSCA-refined) skip
  connection, and fuses two parallel 3×3 convolutions via element-wise
  addition + ReLU, per Eqs. (7)–(11).
- **`models/vasca_net.py`**: wires the above into the full symmetric
  encoder–decoder network shown in Fig. 2(A).
- **`models/ablation.py`**: a togglable variant (`VasCANetAblation`) that
  reproduces every row of Table 1 (Base / Base+A / Base+B / Base+C / …),
  so you can rerun or extend the ablation study.

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

The datasets used in the paper (DRIVE, STARE, CHASE_DB1) require manual
download from their original hosts (registration/license terms vary):

- DRIVE: https://drive.grand-challenge.org/
- STARE: https://cecas.clemson.edu/~ahoover/stare/
- CHASE_DB1: https://blogs.kingston.ac.uk/retinal/chasedb1/

Once downloaded, use `scripts/prepare_data.py` to lay them out into the
folder structure expected by `datasets/retina_dataset.py`:

```
data/DRIVE/
  train/images/*.tif
  train/masks/*.gif
  test/images/*.tif
  test/masks/*.gif
```

Example:

```bash
python scripts/prepare_data.py --dataset drive \
    --raw_dir /path/to/DRIVE --out_dir ./data/DRIVE
```

`RetinalVesselDataset` (see `datasets/retina_dataset.py`) applies the
preprocessing pipeline from Fig. 5 (grayscale → normalisation → CLAHE →
gamma correction), random-crops patches, and applies the paper's
augmentations (random rotation, horizontal/vertical/diagonal flips) during
training.

## Training

```bash
python train.py --config configs/default.yaml
```

Edit `configs/default.yaml` to point `data.root` at your prepared dataset,
and adjust `train.batch_size` / `train.lr` / `train.epochs` as needed
(defaults mirror Section 4.1: batch size 8, Adam @ lr=5e-4, 50 epochs with
early stopping, BCE loss).

## Evaluation

```bash
python test.py --config configs/default.yaml \
    --checkpoint checkpoints/vasca_net_best.pth \
    --save_dir preds/
```

Reports Se / Sp / Precision / F1 / ACC / FPR / AUC / PR-AUC
(`utils/metrics.py`, matching Section 4.3).

## Reproducing the ablation study (Table 1)

```bash
python scripts/run_ablation.py --config configs/default.yaml --epochs 20
```

Trains and evaluates all 8 configurations (Base, +EConv, +DConv, +MSCA,
and combinations) on the same dataset/split, printing a Sp/ACC/AUC/F1 table.

## Repository layout

```
models/
  econv.py        EConv encoder block
  dconv.py        DConv decoder block
  msca.py         Multi-Scale Channel Attention
  vasca_net.py    Full VasCA-Net
  ablation.py     Togglable variant for Table-1-style ablations
datasets/
  retina_dataset.py  Preprocessing + patch dataset for DRIVE/STARE/CHASEDB1
utils/
  losses.py       BCE / Dice / BCE+Dice
  metrics.py      Se, Sp, F1, ACC, AUC, FPR
scripts/
  prepare_data.py Raw-dataset → train/test folder layout converter
  run_ablation.py Table-1 ablation runner
configs/
  default.yaml    Hyperparameters (batch size, lr, epochs, patch size, ...)
train.py          Training loop with early stopping + checkpointing
test.py           Evaluation / inference + optional prediction dump
```

## Implementation notes

A few details in the paper's figures are under-specified or slightly
ambiguous (common with reconstructed architecture diagrams); this repo
makes the following concrete choices, all easy to swap out:

- **EConv reduction ratio / branch widths**: the paper doesn't give exact
  channel numbers per branch, only that both branches are reduced by
  `ratio` and concatenated to `2×Cout/ratio` before a final projection.
  We add an explicit `proj` 1×1 conv to map that back to a configurable
  `out_channels`, which also lets `EConvBlock` double as a channel
  up-projection between encoder stages.
- **MSCA `expand` conv**: to sum the avg- and max-pool branches (which
  live in a reduced `C/ratio` space per kernel scale) and then run a
  single 1×1 conv back to `C` channels for the sigmoid gate, each
  multi-kernel branch re-expands to `C` channels before summation, rather
  than concatenating three different reduced sizes.
- **DConv resolution mismatches**: real fundus images (e.g. DRIVE:
  565×584) aren't powers of two, so `DConvBlock` interpolates the
  upsampled low-level map to exactly match the skip connection's spatial
  size before concatenation.
- **Threshold**: the paper finds 0.5 gives the best overall balance
  (Section 4.4.5, Table 6); this is the default everywhere but is exposed
  as a CLI/config parameter.

If you have access to the authors' original code or supplementary
material, cross-checking `EConvBlock`/`MSCA` channel arithmetic against it
would be the highest-value next step before publishing new results built
on this repo.

## Suggested extensions

- **Sensitivity gap**: the paper notes VasCA-Net's Se is sometimes below
  competing methods (Section 4.5). Try `utils/losses.BCEDiceLoss`, a
  focal-loss variant, or class-balanced sampling of thin-vessel patches.
- **Efficiency**: Section 4.4.10 highlights VasCA-Net's low FLOPs/params;
  try depthwise-separable convolutions in `EConvBlock`/`DConvBlock`, or
  add the pruning/quantization experiments mentioned as future work.
- **Cross-modality**: Section 4.4.3 evaluates on Chest-Xray, ISIC-2018,
  CVC-ClinicDB — `RetinalVesselDataset` is generic enough to point at any
  binary segmentation dataset with an `images/`/`masks/` layout.
- **Multi-modal fusion**: the paper's future work mentions combining
  fundus photography with OCT/OCT-A — a natural place to extend
  `VasCANet.forward` to accept multiple input modalities/branches.

## Citation

```bibtex
@article{ma2026vascanet,
  title   = {VasCA-Net: A vascular channel attention network for retinal vessel segmentation},
  author  = {Ma, Zhendi and Li, Xiaobo and Zhao, Yuxin and Wang, Hui},
  journal = {Expert Systems With Applications},
  volume  = {303},
  pages   = {130591},
  year    = {2026},
  doi     = {10.1016/j.eswa.2025.130591}
}
```
