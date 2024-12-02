#!/bin/sh

#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J painn_job

#BSUB -n 4

#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

#BSUB -W 12:00

#BSUB -o painn_wandb%J.out
#BSUB -e painn_wandb%J.err

source /zhome/51/7/168082/Desktop/s214659/02456_painn_project/painn_env_3/bin/activate

python /zhome/51/7/168082/Desktop/s214659/02456_painn_project/minimal_example_wandb_swa.py
