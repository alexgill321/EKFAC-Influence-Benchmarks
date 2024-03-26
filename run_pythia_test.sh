#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL, END
#SBATCH -o job-%j

set -e  # Exit immediately if a command exits with a non-zero status

# Configurable parameters
COV_BATCH_NUM=${1:-5000}  # Default to 5000 if not specified
MODEL_ID=${2:-"EleutherAI/pythia-12b"}

WORKDIR=$HOME/EKFAC-Influence-Benchmarks
OUTDIR=/scratch/general/vast/$USER/results/pythia-12b
NVIDIA_SMI_LOG=$OUTDIR/nvidia_smi_%j.log
mkdir -p $OUTDIR

echo "Starting job at $(date)"

nohup watch -n 10 "nvidia-smi >> $NVIDIA_SMI_LOG" &
source $WORKDIR/ekfac/bin/activate
python $WORKDIR/src/pythia_test.py --pile_dir $WORKDIR --ekfac_dir $WORKDIR --cov_batch_num 5000 --output_dir $OUTDIR --model_id "EleutherAI/pythia-12b"

kill $!