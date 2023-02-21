#!/bin/bash
#SBATCH --job-name=mp_test
#SBATCH --gres=gpu:2
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=5
# SBATCH --time=0-4:00:00
#SBATCH --mem=16GB
#SBATCH --qos=millerlab
#SBATCH --partition=millerlab
#SBATCH --output='mp_test_out'
##SBATCH --output=/om2/user/eisenaj/code/jupyter/jupyter_logs/dldmd-%j.out

# nvidia-smi

cd /om2/user/eisenaj/code/DeLASE
unset XDG_RUNTIME_DIR
source activate dynamical-trajectories
python mp_testing.py