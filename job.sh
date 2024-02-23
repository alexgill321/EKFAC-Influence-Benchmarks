#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#
#SBATCH --ntasks=1
#SBATCH -o job-%j
#SBATCH --time=2:00:00

WORKDIR=$HOME/EKFAC-Influence-Benchmarks
SCRDIR=/scratch/general/vast/$USER/pile
mkdir -p ${SCRDIR}

source $WORKDIR/ekfac/bin/activate

python $WORKDIR/download_pile_chpc.py --output_dir $SCRDIR

