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

def add_offline_eval_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title='ModelOpt text generation ptq')
    group.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    group.add_argument(
        "--offline-output-dir",
        type=str,
        required=True,
    )
    group.add_argument(
        "--offline-batch-size", type=int, default=4, help="Batch size to use for ptq calibration."
    )
    add_modelopt_args(parser)
    return parser

def load_datasets(dataset_dir):
    dataset_dir = Path(dataset_dir)
    prompt_files = list(dataset_dir.glob("*.json"))
    datasets = {}
    for prompt_file in prompt_files:
        with open(prompt_file, "r") as fin:
            dataset = json.load(fin)


        datasets[dataset['target_dataset']] = dataset
    return datasets


def data_loader(dataset, batch_size):
    dataset_size = max(len(dataset), batch_size)
    for i in range(dataset_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        yield batch


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_offline_eval_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()

    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text generation.")
    args.exit_on_missing_checkpoint = True

    datasets = load_datasets(args.dataset_dir)

    # Set up model and load checkpoint
    # [ModelOpt]: make sure that output logits are allgathered.
    text_generation_model_provider = functools.partial(model_provider, parallel_output=False)
    model = get_model(text_generation_model_provider, wrap_with_ddp=False)

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        # _ = load_checkpoint(model, None, None)
        print_rank_0("Done loading checkpoint")

    def custom_prompt_forward_loop_func(model):
        for name, dataset in datasets.items():
            output_path = os.path.join(args.offline_output_dir, f"{name}.eval-generated.json")
            if os.path.exists(output_path):
                continue

            batchs = list(data_loader(dataset['samples'], args.offline_batch_size))
            for batch in tqdm(batchs, desc=name):
                prompts = [sample['prompt'] for sample in batch]
                if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                    (
                        prompts_plus_generations,
                        prompts_plus_generations_segments,
                        logprobs,
                        _,
                    ) = generate_and_post_process(
                        model,
                        prompts=prompts,
                        tokens_to_generate=dataset['output_length'],
                        return_output_log_probs=False,
                        top_k_sampling=1,
                        add_BOS=True,
                        temperature=1.0,
                    )
                    
                    for sample, prompt in zip(batch, prompts_plus_generations):
                        sample["generated"] = prompt
                else:
                    generate_and_post_process(model)
            
            if mpu.get_tensor_model_parallel_rank() == 0:
                os.makedirs(args.offline_output_dir, exist_ok=True)
                with open(output_path, "w") as fout:
                    json.dump(dataset, fout, indent=4, ensure_ascii=False)

    # Setting data parallel and tensor parallel group
    set_data_parallel_group(mpu.get_data_parallel_group())
    set_tensor_parallel_group(mpu.get_tensor_model_parallel_group())

    custom_prompt_forward_loop_func(model[0])

