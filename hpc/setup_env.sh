#!/bin/bash
# Run this ONCE on HPC login node to set up the conda environment
# Usage: cd /scratch/lustre/home/$USER/Kursinis && bash hpc/setup_env.sh

echo "=================================================="
echo "Setting up Conda environment for HPC GPU training"
echo "=================================================="
echo "Current directory: $(pwd)"

# Activate miniconda
source ~/miniconda3/bin/activate

echo ""
echo "[1/4] Creating conda environment 'kursinis'..."
conda create -n kursinis python=3.10 -y

echo ""
echo "[2/4] Activating environment..."
conda activate kursinis

echo ""
echo "[3/4] Installing PyTorch with CUDA and ML dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-metric-learning
pip install albumentations
pip install scikit-learn
pip install scipy
pip install Pillow
pip install tqdm

echo ""
echo "[4/4] Creating directories..."
mkdir -p logs checkpoints results

echo ""
echo "=================================================="
echo "Environment setup complete!"
echo "=================================================="
echo ""
echo "To test (submit an interactive GPU job):"
echo "  srun -p gpu --gres gpu --pty bash"
echo "  source ~/miniconda3/bin/activate && conda activate kursinis"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"

