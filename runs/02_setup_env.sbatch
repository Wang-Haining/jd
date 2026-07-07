#!/bin/bash
#SBATCH --job-name=jd_setup
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# ============================================================================
# Step 1: Create venv and install all dependencies
#
# USAGE:
#   sbatch runs/02_setup_env.sbatch
#
# BEFORE RUNNING: ensure runs/01_check_gpu.sbatch passed
# ============================================================================

set -euo pipefail

module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

PROJ=/home/g91p721/jd
cd $PROJ

python -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1

pip install --upgrade pip

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core deps
pip install transformers accelerate datasets
pip install scipy scikit-learn matplotlib seaborn pandas statsmodels
pip install tqdm pyyaml openai pytest

# J-lens — try pip name first, fall back to git
pip install jlens 2>/dev/null || \
pip install jacobian-lens 2>/dev/null || \
pip install git+https://github.com/anthropics/jacobian-lens.git

# Verify all imports
python -c "
import jlens
import torch
import transformers
from datasets import load_dataset
print('=== ALL IMPORTS OK ===')
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'transformers {transformers.__version__}')
print(f'jlens: {dir(jlens)}')
"

echo "=== SETUP COMPLETE ==="
