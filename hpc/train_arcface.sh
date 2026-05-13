#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_arcface_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_arcface_%j.err
#SBATCH --job-name=vits_arcface

cd /scratch/lustre/home/$USER/Kursinis

source ~/miniconda3/bin/activate
conda activate kursinis

echo "Starting ViT-S + ArcFace Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

export PYTHONUNBUFFERED=1

python3 -u src/train.py \
    --backbone vits \
    --loss arcface \
    --epochs 20 \
    --batch-size 1024 \
    --lr 0.0004 \
    --checkpoint-minutes 30 \
    --eval-epochs 5 \
    --workers 4

echo "Training complete!"
