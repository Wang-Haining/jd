#!/bin/bash
#SBATCH --job-name=jd_gpu_check
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.err

# ============================================================================
# Step 0: Verify GPU availability on Tempest
#
# USAGE:
#   sbatch runs/01_check_gpu.sbatch
#
# IF THIS FAILS with "Invalid partition" or similar:
#   sinfo -o "%P %G"   # list all partitions and their GPUs
#   # Try alternative partition names: gpuA100, gpu-a100, nvidia
#   # If no GPU partition exists, fall back to IU Quartz
# ============================================================================

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh

echo "=== GPU CHECK ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo ""

nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

python3 --version
python3 -c "import sys; print(f'Python executable: {sys.executable}')"

echo ""
echo "=== GPU CHECK COMPLETE ==="
