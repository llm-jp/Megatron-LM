# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
from pathlib import Path

os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"
os.environ[
        "NCCL_DEBUG"
] = "DEBUG"
os.environ[
        "NCCL_ASYNC_ERROR_HANDLING"
] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.utils.distributed import set_data_parallel_group, set_tensor_parallel_group
from tqdm import tqdm

print(os.getcwd())

# [ModelOpt]: changing the default model provider to the ModelOpt version
from megatron.core import mpu
from megatron.inference.arguments import add_modelopt_args
from megatron.inference.checkpointing import load_modelopt_checkpoint
from megatron.inference.gpt.model_provider import model_provider
from megatron.inference.text_generation import generate_and_post_process
from megatron.training import get_args, get_model, initialize_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model
from megatron.training.checkpointing import load_checkpoint

QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
}


def add_trtllm_ckpt_export_args(parser):
    """Add additional arguments for TensorRT-LLM."""
    group = parser.add_argument_group(title="trtllm")

    group.add_argument(
        "--export-dir", type=str, help="The output TensorRT-LLM checkpoint.",
    )
    group.add_argument(
        "--decoder", type=str, choices=["gptnext", 'llama'], help="The decoder type of the model.",
    )
    group.add_argument(
        "--inference-tensor-parallel",
        type=int,
        help="Tensor parallel for the inference time, can be different from the training config.",
        default=1,
    )


def add_text_generate_ptq_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title='ModelOpt text generation ptq')
    group.add_argument(
        "--calib-dataset",
        type=str,
        default="cnn_dailymail",
        help="Calibration datasets from HuggingFace datasets.",
    )
    group.add_argument(
        "--calib-batch-size", type=int, default=4, help="Batch size to use for ptq calibration."
    )
    group.add_argument(
        "--calib-size", type=int, default=512, help="Samples to use for ptq calibration."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    add_modelopt_args(parser)
    add_trtllm_ckpt_export_args(parser)
    return parser


def get_calib_dataloader(
    data="cnn_dailymail", batch_size=4, calib_size=512, max_sequence_length=512
):
    if data == "pileval":
        dataset = load_dataset(
            "json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train"
        )
        text_column = "text"
    elif data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"

    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_ptq_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    print("ModelOpt PTQ Text Generation")

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    # [ModelOpt]: make sure that output logits are allgathered.
    text_generation_model_provider = functools.partial(model_provider, parallel_output=False)
    model = get_model(text_generation_model_provider, wrap_with_ddp=False)

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        # _ = load_checkpoint(model, None, None)
        print_rank_0("Done loading checkpoint")

    # Removing virtual pipeline parallel and other wrapper
    assert len(model) == 1, "Above condition should have caught this"
    unwrapped_model = unwrap_model(model)

    all_prompts = args.prompts.split("|")

    def custom_prompt_forward_loop_func(model):
        for prompt in tqdm(all_prompts):
            if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                (
                    prompts_plus_generations,
                    prompts_plus_generations_segments,
                    logprobs,
                    _,
                ) = generate_and_post_process(
                    model,
                    prompts=[prompt],
                    tokens_to_generate=128,
                    return_output_log_probs=True,
                    top_k_sampling=1,
                    add_BOS=True,
                    temperature=1.0,
                )
                print_rank_0(prompts_plus_generations[0])
            else:
                generate_and_post_process(model)

    # Setting data parallel and tensor parallel group
    set_data_parallel_group(mpu.get_data_parallel_group())
    set_tensor_parallel_group(mpu.get_tensor_model_parallel_group())

    if args.save is not None and args.export_quant_cfg in QUANT_CFG_CHOICES:
        save_checkpoint(1, unwrapped_model, None, None, 0)

    custom_prompt_forward_loop_func(model[0])

