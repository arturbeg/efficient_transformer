#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=moe_transformer_cuda

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
#SBATCH --output=moe_transformer_cuda_%j.out

# run the application
srun python main.py --gating moe --optimizer adam --num-epochs 2 --bsz 64 --bptt 128 --d-model 256 --lr 1.0 --n-heads 4 --decoder-mixing none --num-experts 4 --k 2 --cuda --sparse_attn False