# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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

from megatron.training.utils import print_rank_0

def add_inference_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title='Text generation')
    group.add_argument(
        "--prompts",
        type=str,
        required=True,
    )
    add_modelopt_args(parser)
    return parser


def data_loader(dataset, batch_size):
    dataset_size = max(len(dataset), batch_size)
    for i in range(dataset_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        yield batch


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_inference_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()

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
                    temperature=1.0,
                )
                print_rank_0(prompts_plus_generations[0])
            else:
                generate_and_post_process(model)

    # Setting data parallel and tensor parallel group
    set_data_parallel_group(mpu.get_data_parallel_group())
    set_tensor_parallel_group(mpu.get_tensor_model_parallel_group())

    custom_prompt_forward_loop_func(model[0])

