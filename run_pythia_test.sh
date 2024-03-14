#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o job-%j

WORKDIR=$HOME/EKFAC-Influence-Benchmarks
OUTDIR=/scratch/general/vast/$USER/results
NVIDIA_SMI_LOG=$OUTDIR/nvidia_smi.log

nohup  "nvidia-smi -l 30 -f $NVIDIA_SMI_LOG" &
source $WORKDIR/ekfac/bin/activate
python $WORKDIR/src/pythia_test.py --pile_dir $WORKDIR --ekfac_dir $WORKDIR --cov_batch_num 5000 --output_dir $OUTDIR