#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=245GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o job-%j

WORKDIR=$HOME/EKFAC-Influence-Benchmarks
OUTDIR=/scratch/general/vast/$USER/results/pythia-70m
NVIDIA_SMI_LOG=$OUTDIR/nvidia_smi.log
mkdir -p $OUTDIR

nohup watch -n 10 "nvidia-smi >> $NVIDIA_SMI_LOG" &
source $WORKDIR/ekfac/bin/activate
python $WORKDIR/evals/pythia_kronfluence.py --data_dir $WORKDIR --ekfac_dir $WORKDIR --cov_batch_num 200 --output_dir $OUTDIR --model_id "EleutherAI/pythia-70m" --query_batch_size 32 --train_batch_size 32