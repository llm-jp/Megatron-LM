<<COMMENT
#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llama/Megatron-LM
COMMENT

deactivate
rm -r .env
python3 -m venv .env
source .env/bin/activate

pip install --upgrade pip
pip install --no-cache-dir ninja wheel packaging
pip install --no-cache-dir setuptools==69.5.1
pip install --no-cache-dir -r requirements.txt

pip install  --no-cache-dir datasets flask_restful

# huggingface install
pip install  --no-cache-dir accelerate zarr tensorstore
pip install  --no-cache-dir transformers==4.40.2

# transformer engine
pip install --no-cache-dir git+https://github.com/NVIDIA/TransformerEngine.git@c81733f1032a56a817b594c8971a738108ded7d0

# flash-attention install
pip install flash-attn==2.4.2 --no-cache-dir
# pip install flash-attn --no-build-isolation

# pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-llm

# apex install
rm -rf apex
git clone https://github.com/NVIDIA/apex.git
cd apex/
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ../

pip install "nvidia-modelopt[all]~=0.13.0" --extra-index-url https://pypi.nvidia.com --no-cache-dir

bash examples/inference/quantization/ptq_trtllm_llama_7b.sh /model/kouta/model/megatron/llm-jp-13B-v3
