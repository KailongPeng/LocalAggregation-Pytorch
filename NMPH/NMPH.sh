#!/bin/bash
#SBATCH -p psych_day  # psych_gpu  # psych_day,psych_gpu,psych_scavenge,psych_week， psych_scavenge
#SBATCH --job-name=NMPH
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out  # %J.out  # %A_%a.out
#SBATCH --mem=50g
# #SBATCH --gpus 1

set -e
# nvidia-smi
cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate py36_jupyter  # py36_jupyter
python -u ./NMPH/NMPH.py "${SLURM_ARRAY_TASK_ID}"

echo "done"

