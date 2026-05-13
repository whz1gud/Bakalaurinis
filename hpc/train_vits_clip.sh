#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_vits_clip_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_vits_clip_%j.err
#SBATCH --job-name=vits_clip

cd /scratch/lustre/home/$USER/Kursinis

source ~/miniconda3/bin/activate
conda activate kursinis

echo "Starting ViT-S + CLIP/InfoNCE Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

export PYTHONUNBUFFERED=1

# CLIP/InfoNCE uses PairDataset (2 views per image), so effective batch is 2x
# Batch 256 → loads 512 images through model per step
export CLIP_TEMPERATURE=0.07
python3 -u src/train.py \
    --backbone vits \
    --loss clip \
    --epochs 20 \
    --batch-size 256 \
    --lr 0.0001 \
    --checkpoint-minutes 30 \
    --eval-epochs 5 \
    --workers 4

echo "Training complete!"
