# Vision Transformer Fine-Tuning for Image Similarity Using Metric Learning

Bachelor's thesis comparing five metric learning loss functions for fine-tuning DINOv2 on the DISC21 image similarity dataset, across two backbone sizes (ViT-S/14 and ViT-B/14).

## Overview

This project evaluates five metric learning loss functions for image copy detection:

- **Contrastive Loss** — pairwise distance-based learning
- **Triplet Loss** — anchor-positive-negative triplet learning
- **ArcFace Loss** — angular margin-based classification
- **Focal Loss** — hard-example weighted cross-entropy
- **CLIP/InfoNCE Loss** — batch-level contrastive learning with temperature scaling

All methods fine-tune a pre-trained DINOv2 model on the DISC21 dataset. Both ViT-S/14 (21M parameters) and ViT-B/14 (86M parameters) backbones are evaluated, for a total of 12 experimental configurations.

## Results

CLIP/InfoNCE achieved the best performance on both backbones:

| Configuration | P@1 | mAP | μAP |
|---|---|---|---|
| ViT-S Baseline | 56.71% | 58.80% | 41.44% |
| ViT-S Contrastive | 55.39% | 57.90% | 47.90% |
| ViT-S Triplet | 55.58% | 58.53% | 42.14% |
| ViT-S ArcFace | 58.41% | 60.41% | 48.82% |
| ViT-S Focal | 56.52% | 58.64% | 43.93% |
| **ViT-S CLIP** | **60.49%** | **63.13%** | **48.80%** |
| ViT-B Baseline | 57.84% | 60.93% | 39.92% |
| ViT-B Contrastive | 60.30% | 62.24% | 51.37% |
| ViT-B Triplet | 58.41% | 61.27% | 42.15% |
| ViT-B ArcFace | 59.17% | 61.43% | 43.27% |
| ViT-B Focal | 58.79% | 61.37% | 45.86% |
| **ViT-B CLIP** | **61.63%** | **64.11%** | **49.77%** |

See `bachelor_thesis_template_vu_mif_se/bakalaurinis.pdf` for the full thesis with analysis.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data

Download the DISC21 dataset and extract to `data/`:

```
data/train/          - Training images
data/queries_dev/    - Development query images
data/queries_test/   - Test query images
data/refs/           - Reference database images
```

## Usage

### Training

```bash
# ViT-S/14 (default backbone)
python src/train.py --loss contrastive --epochs 20 --batch-size 256
python src/train.py --loss triplet     --epochs 20 --batch-size 256
python src/train.py --loss arcface     --epochs 20 --batch-size 256
python src/train.py --loss focal       --epochs 20 --batch-size 256
python src/train.py --loss clip        --epochs 20 --batch-size 256

# ViT-B/14 backbone
python src/train.py --loss clip --backbone vitb --epochs 20 --batch-size 128
```

### Evaluation

```bash
# Evaluate a trained model
python src/evaluate.py --checkpoint checkpoints/clip_best.pt

# Run baseline (no fine-tuning)
python src/run_baseline.py
```

### HPC Training

For HPC cluster (SLURM) usage, see `hpc/README_HPC.md`.

```bash
# ViT-S jobs
sbatch hpc/train_contrastive.sh
sbatch hpc/train_triplet.sh
sbatch hpc/train_arcface.sh
sbatch hpc/train_vits_focal.sh
sbatch hpc/train_vits_clip.sh

# ViT-B jobs
sbatch hpc/train_vitb_contrastive.sh
sbatch hpc/train_vitb_arcface.sh
sbatch hpc/train_vitb_focal.sh
sbatch hpc/train_vitb_clip.sh
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Evaluation script
│   ├── run_baseline.py     # Baseline evaluation
│   ├── visualize_embeddings.py
│   ├── data/               # Dataset classes
│   ├── models/             # Model architectures
│   └── evaluation/         # Evaluation metrics
├── hpc/                    # HPC SLURM scripts
├── logs/                   # Training logs and per-epoch history
├── results/                # Evaluation result JSON files
├── visualizations/         # Embedding space figures used in the thesis
└── bachelor_thesis_template_vu_mif_se/
    ├── bakalaurinis.tex    # Thesis LaTeX source
    ├── bakalaurinis.pdf    # Compiled thesis PDF
    └── bibliografija.bib   # Bibliography
```

> **Note:** Model checkpoints and the DISC21 dataset are not included in this repository due to size. Checkpoints are available on request; the dataset can be downloaded from the [DISC21 challenge page](https://github.com/facebookresearch/isc2021).

## Citation

If you use this code, please cite:

- DISC21 Dataset: Douze et al. (2021)
- DINOv2: Oquab et al. (2024)
- ArcFace: Deng et al. (2022)
- CLIP/InfoNCE: Radford et al. (2021), van den Oord et al. (2018)

Full citations are in `bachelor_thesis_template_vu_mif_se/bibliografija.bib`.

## License

This project is submitted as a bachelor's thesis at Vilnius University, Faculty of Mathematics and Informatics.
