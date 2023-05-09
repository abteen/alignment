#!/bin/bash

#Run with 'sbatch --export=ALL,seed=42 finetune_xlmr_large.slurm'

#SBATCH --ntasks=24
#SBATCH --mem=32g
#SBATCH --time=2-00:00:00
#SBATCh --partition=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu:1
#SBATCH -J layer_analysis_test



module purge
module load python

cd /projects/abeb4417/

source startup.sh

cd alignment/xlmr-awesome/

./scripts/all_layer_analysis.sh