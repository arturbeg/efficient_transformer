#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=moe_ffn_cuda

# set number of GPUs
#SBATCH --gres=gpu:1

# set number of CPUs
#SBATCH --cpus-per-task=5

# set partition
#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=artur.begyan@kellogg.ox.ac.uk

# Standard output and error log
#SBATCH --output=moe_ffn_cuda_%j.out

#SBATCH --array=0-2

# run the application

ARGS=(4 16 64)

srun python main.py --cuda --n-layers 3 --log-and-save-file-name moe_ffn_$SLURM_ARRAY_TASK_ID --optimizer adam --num-epochs 10 --bsz 64 --bptt 128 --d-model 256 --lr 1.0 --n-heads 4 --decoder-mixing none --ff-gating moe_gshard --gating none --num-experts ${ARGS[$SLURM_ARRAY_TASK_ID]} --k 2 --sparse_attn False --debug False