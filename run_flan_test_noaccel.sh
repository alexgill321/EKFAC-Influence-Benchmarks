#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o job-%j

WORKDIR=$HOME/EKFAC-Influence-Benchmarks
OUTDIR=/scratch/general/vast/$USER/results/flan-t5-xl
MODELDIR=/scratch/general/vast/u1420010/final_models/output_model/contract-nli/checkpoint-300/
DATADIR=/scratch/general/vast/u1420010/final_models/data
NVIDIA_SMI_LOG=$OUTDIR/nvidia_smi_flan.log
# layers=("decoder.block.3.layer.2.DenseReluDense.wi_0" "encoder.block.4.layer.1.DenseReluDense.wi_0")
# layerArray="${layers[*]}"

mkdir -p $OUTDIR
nohup watch -n 10 "nvidia-smi >> $NVIDIA_SMI_LOG" &
source $WORKDIR/ekfac/bin/activate
python $WORKDIR/src/flan_t5_test_noaccel.py --data_dir $DATADIR --ekfac_dir $WORKDIR --cov_batch_num 100 --output_dir $OUTDIR --model_dir $MODELDIR --test_start_idx 0 --test_end_idx 100
