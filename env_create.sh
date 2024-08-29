mamba create -n DeLASE -y python=3.11
source activate DeLASE
mamba install -y jupyter jupyterlab matplotlib numpy scipy tqdm pip cython -c conda-forge
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install sdeint
python -m pip install --editable .