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

#SBATCH --array=0-7

# run the application

ARGS=(8 16 32 64 128 256 512, 1024)

srun python main.py --gating none --optimizer adam --num-epochs 2 --bsz 64 --bptt 128 --d-model 256 --lr 1.0 --n-heads 4 --decoder-mixing none --ff-gating moe --num-experts ${ARGS[$SLURM_ARRAY_TASK_ID]} --k 2 --cuda