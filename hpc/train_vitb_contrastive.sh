#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_vitb_contrastive_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_vitb_contrastive_%j.err
#SBATCH --job-name=vitb_contrastive

cd /scratch/lustre/home/$USER/Kursinis

source ~/miniconda3/bin/activate
conda activate kursinis

echo "Starting ViT-B + Contrastive Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

export PYTHONUNBUFFERED=1

# ViT-B: batch 256 for contrastive (same ratio as ViT-S which used 256)
python3 -u src/train.py \
    --backbone vitb \
    --loss contrastive \
    --epochs 20 \
    --batch-size 256 \
    --lr 0.0001 \
    --checkpoint-minutes 30 \
    --eval-epochs 5 \
    --workers 4

echo "Training complete!"
