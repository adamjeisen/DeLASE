#!/bin/bash
#SBATCH --job-name=mp_delase
#SBATCH --gres=gpu:2
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=5
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --qos=millerlab
#SBATCH --partition=millerlab
##SBATCH --output='mp_delase'
#SBATCH --output=/om2/user/eisenaj/code/DeLASE/mp_delase-%j
#SBATCH --mail-user=eisenaj@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# nvidia-smi

cd /om2/user/eisenaj/code/DeLASE
unset XDG_RUNTIME_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/om2/user/eisenaj/anaconda/lib
source activate dynamical-trajectories
python mp_delase.py