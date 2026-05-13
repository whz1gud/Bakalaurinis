#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_vitb_focal_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_vitb_focal_%j.err
#SBATCH --job-name=vitb_focal

cd /scratch/lustre/home/$USER/Kursinis

source ~/miniconda3/bin/activate
conda activate kursinis

echo "Starting ViT-B + Focal Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

export PYTHONUNBUFFERED=1

# ViT-B: batch 512 (reduced from ViT-S's 1024 due to larger model)
# LR scaled proportionally: 0.0004 * (512/1024) = 0.0002
python3 -u src/train.py \
    --backbone vitb \
    --loss focal \
    --epochs 20 \
    --batch-size 512 \
    --lr 0.0002 \
    --checkpoint-minutes 30 \
    --eval-epochs 5 \
    --workers 4

echo "Training complete!"
