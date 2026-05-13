#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_baseline_vitb_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_baseline_vitb_%j.err
#SBATCH --job-name=baseline_vitb

cd /scratch/lustre/home/$USER/Kursinis

source ~/miniconda3/bin/activate
conda activate kursinis

echo "Running ViT-B Baseline Evaluation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi

export PYTHONUNBUFFERED=1
python3 -u src/run_baseline.py --backbone vitb --batch-size 128 --workers 4

echo "Baseline evaluation complete!"
