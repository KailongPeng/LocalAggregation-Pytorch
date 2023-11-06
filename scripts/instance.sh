#!/bin/bash
#SBATCH -p psych_gpu  # psych_day,psych_gpu,psych_scavenge,psych_week， psych_scavenge
#SBATCH --job-name=LA
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=100g
#SBATCH --gpus 1

set -e
nvidia-smi
cd /gpfs/milgram/project/turk-browne/projects/sandbox/sandbox/MNIST
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate py36

python --version
python -u /gpfs/milgram/pi/turk-browne/projects/sandbox/sandbox/docker/hello.py

CUDA_VISIBLE_DEVICES=0 python -u ./scripts/instance.py

nvidia-smi

echo "done"
