#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# priotiry: -5: normal, -4: high, -3: highest

# Load modules
module load cuda/12.1.0
module load nccl/2.20.5
module load openmpi/5.0.2-gcc
module load ninja/1.11.1
module load ~/modulefiles/cudnn/9.0.0

# Set environment variables
source .env/bin/activate

pip install --upgrade pip

# Install packages
pip install -r requirements.txt

# nvidia apex
cd ..

git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# flash-atten
cd ..
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention

python setup.py install
