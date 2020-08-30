#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=sparse_attn

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
#SBATCH --output=sparse_attn_%j.out

# run the application

#SBATCH --array=0-3

ARGS=(128 256 512)

srun python main.py --cuda --n-layers 3 --log-and-save-file-name sparse_attn --optimizer adam --num-epochs 10 --bsz 64 --bptt ${ARGS[$SLURM_ARRAY_TASK_ID]} --d-model 512 --lr 1.0 --n-heads 8 --decoder-mixing none --ff-gating none --gating none --sparse_attn