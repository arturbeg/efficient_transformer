#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=sparse_attn_1024

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
#SBATCH --output=sparse_attn_1024_%j.out

#SBATCH --array=0-4

# run the application

ARGS=(4 8 16 32)

srun python main.py --cuda --n-layers 3 --log-and-save-file-name sparse_attn_1024_$SLURM_ARRAY_TASK_ID --optimizer adam --num-epochs 10 --bsz 64 --bptt 512 --d-model 256 --lr 1.0 --n-heads 8 --decoder-mixing none --ff-gating none --gating none --sparse_attn --num-sub ${ARGS[$SLURM_ARRAY_TASK_ID]}