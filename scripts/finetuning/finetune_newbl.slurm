#!/bin/bash

#Run with 'sbatch --export=ALL,seed=42 finetune_xlmr_large.slurm'

#SBATCH --ntasks=24
#SBATCH --mem=32g
#SBATCH --time=1-00:00:00
#SBATCh --partition=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu:v100:1
#SBATCH -J pos_fa_base



module purge
module load python

cd /projects/abeb4417/

source startup.sh
conda activate torch_a100

cd alignment/

./scripts/finetuning/finetune_pos.sh gn gn fastalign bert-base-multilingual-cased