#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_vits_focal_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_vits_focal_%j.err
#SBATCH --job-name=vits_focal

cd /scratch/lustre/home/$USER/Kursinis

source ~/miniconda3/bin/activate
conda activate kursinis

echo "Starting ViT-S + Focal Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

export PYTHONUNBUFFERED=1

# Same batch/lr as ArcFace on ViT-S (classification-based, similar structure)
python3 -u src/train.py \
    --backbone vits \
    --loss focal \
    --epochs 20 \
    --batch-size 1024 \
    --lr 0.0004 \
    --checkpoint-minutes 30 \
    --eval-epochs 5 \
    --workers 4

echo "Training complete!"
