#/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_BASE=$(conda info --base)
conda env remove -n env
conda create -n env python=3.9 -y
conda activate env
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu111
./launch_server