#!/bin/bash
# Shared Tempest module/cache setup for jd Slurm jobs.

source /usr/local/Modules/5.6.1/init/bash
module purge
source /home/g91p721/.module/py312

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/g91p721/hf-cache
