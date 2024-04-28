import os
import subprocess
import tempfile

import pytest
from transformers import AutoModel

HF_LLAMA_8B_PATH = "/home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Meta-Llama-3-8B"


@pytest.mark.parametrize("tp", ["1", "2"])
@pytest.mark.parametrize("pp", ["1", "2"])
@pytest.mark.parametrize("hf_path", [HF_LLAMA_8B_PATH])
def test_llama_2_hf_megatron_hf(tp, pp, hf_path):
    with tempfile.TemporaryDirectory(
        dir="/home/ext_kazuki_fujii_rio_gsic_titech/tmp/megatron"
    ) as temp_dir_meg, tempfile.TemporaryDirectory(
        dir="/home/ext_kazuki_fujii_rio_gsic_titech/tmp/hf"
    ) as temp_dir_hf:
        print('STEP 1: convert: hf -> megatron', flush=True)
        cmd = [
            "python",
            "tools/checkpoint/convert.py",
            "--model-type", "GPT",
            "--loader", "llama3_hf",
            "--saver", "mcore",
            "--target-tensor-parallel-size", tp,
            "--target-pipeline-parallel-size", pp,
            "--tokenizer-model", hf_path,
            "--load-dir", hf_path,
            "--save-dir", temp_dir_meg,
            "--bf16",
            "--saver-transformer-impl", "transformer_engine",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        print('STEP 2: convert: megatron -> hf', flush=True)
        cmd = [
            "python",
            "tools/checkpoint/convert.py",
            "--model-type", "GPT",
            "--loader", "mcore",
            "--saver", "llama3_hf",
            "--load-dir", temp_dir_meg,
            "--save-dir", temp_dir_hf,
            "--hf-tokenizer-path", hf_path,
            "--save-dtype", "bfloat16",
            "--loader-transformer-impl", "transformer_engine",
            "--megatron-path", "/home/ext_kazuki_fujii_rio_gsic_titech/src/Megatron-LM",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        print('CHECK: hf -> megatron -> hf', flush=True)
        model1 = AutoModel.from_pretrained(hf_path)
        model2 = AutoModel.from_pretrained(temp_dir_hf)

        for t1, t2 in zip(model1.named_buffers(), model2.named_buffers()):
            name1, param1 = t1
            name2, param2 = t2
            assert name1 == name2, "buffer name must match"
            assert param1.data.ne(param2.data).sum() == 0, f"{name1} seems to be wrong"

        for t1, t2 in zip(model1.named_parameters(), model2.named_parameters()):
            name1, param1 = t1
            name2, param2 = t2
            assert name1 == name2, "parameter name must match"
            assert param1.data.ne(param2.data).sum() == 0, f"{name1} seems to be wrong"
