#!/bin/bash
#SBATCH --job-name=jd_gpu_check
#SBATCH --partition=gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
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

echo "=== GPU CHECK ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo ""

nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Memory: {props.total_mem / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
    # Can we fit a 7B fp16 model? Need ~14GB inference, ~28GB for J-lens fitting
    if props.total_mem / 1e9 >= 24:
        print('STATUS: OK for 7B model J-lens fitting')
    elif props.total_mem / 1e9 >= 16:
        print('STATUS: Tight — use 3B model or 4-bit quantization for 7B')
    else:
        print('STATUS: Insufficient — fall back to Quartz A100s')
else:
    print('STATUS: NO GPU — fall back to Quartz')
"

echo ""
echo "=== GPU CHECK COMPLETE ==="
