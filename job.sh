#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#
#SBATCH --ntasks=1
#SBATCH -o job-%j
#SBATCH --time=2:00:00

WORKDIR=$HOME/pythia
VENVDIR=$HOME/EKFAC-Influence-Benchmarks
SCRDIR=/scratch/general/vast/$USER/pile
INPUTDIR=$SCRDIR/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/bac79b6820adb34e451f9a02cc1dc7cd920febf0/document-00000-of-00020.bin

source $VENVDIR/ekfac/bin/activate

mkdir -p $SCRDIR/small

python $WORKDIR/utils/unshard_memmap.py --input_file $INPUTDIR --num_shards 1 --output_dir $SCRDIR/small

