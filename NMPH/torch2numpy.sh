#!/bin/bash
#SBATCH -p psych_day  # psych_gpu  # psych_day,psych_gpu,psych_scavenge,psych_week， psych_scavenge
#SBATCH --job-name=torch2numpy
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out  # %A_%a.out
#SBATCH --mem=100g
##SBATCH --gpus 1

set -e
#nvidia-smi
cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate py36
python -u ./NMPH/torch2numpy.py

echo "done"

