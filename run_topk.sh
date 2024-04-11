#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o job-%j

DATADIR=/scratch/general/vast/u1420010/final_models/data
OUTDIR=/scratch/general/vast/$USER/results
INFDIR=$OUTDIR/flan-t5-xl

WORKDIR=$HOME/EKFAC-Influence-Benchmarks

source $WORKDIR/ekfac/bin/activate

python $WORKDIR/src/k_top_scores.py --data_dir $DATADIR --influence_scores_dir $INFDIR --output_dir $OUTDIR
