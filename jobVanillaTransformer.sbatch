#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vanilla_transformer_4_layers

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
#SBATCH --output=vanilla_transformer_4_layers_%j.out

srun python main.py --cuda --n-layers 4 --log-and-save-file-name vanilla_transformer_4_layers --optimizer adam --num-epochs 10 --bsz 128 --bptt 128 --d-model 256 --lr 1.0 --n-heads 8 --decoder-mixing none --ff-gating none --gating none