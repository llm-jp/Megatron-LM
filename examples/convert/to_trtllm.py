# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pathlib import Path

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training.utils import unwrap_model
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.gpt import GPTModel
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.inference.text_generation_server import MegatronServer
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.inference.checkpointing import load_modelopt_checkpoint
from megatron.inference.arguments import add_modelopt_args
from megatron.training.checkpointing import save_checkpoint

from modelopt.torch.export import export_tensorrt_llm_checkpoint

import torch
from typing import Union
import megatron

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the core GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process
        )
    else:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )

    return model

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

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')

    add_modelopt_args(parser)
    add_trtllm_ckpt_export_args(parser)
    return parser

if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        # _ = load_checkpoint(model, None, None)

    unwrapped_model = unwrap_model(model)

    export_tensorrt_llm_checkpoint(
        unwrapped_model[0],
        args.decoder,
        torch.bfloat16 if args.bf16 else torch.float16,
        export_dir=args.export_dir,
        inference_tensor_parallel=args.inference_tensor_parallel,
        inference_pipeline_parallel=1,
        use_nfs_workspace=True,
    )