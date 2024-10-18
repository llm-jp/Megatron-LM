import sys
from unittest.mock import Mock

# Avoid importing `transformer_engine` as it checks for the
# presense of CUDA and thus fails on CPU-only systems.
sys.modules["transformer_engine"] = Mock()

import json
import os
from pathlib import Path
from typing import Dict, Union

import numpy
import torch
from pretrain_gpt import train_valid_test_datasets_provider
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.gpt_dataset import GPTDataset, _get_ltor_masks_and_position_ids
from megatron.training.tokenizer.tokenizer import build_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.training import (
    get_args,
    build_train_valid_test_datasets,
)
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler


def _add_train_data_dumping_args(parser):
    group = parser.add_argument_group(title='train data dumping')
    group.add_argument(
        '--train-data-dump-path',
        type=str,
        required=True,
        help='Path to dump the training data',
    )
    return parser


def blended_dataset_getitem(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
    dataset_id = self.dataset_index[idx]
    dataset_sample_id = self.dataset_sample_index[idx]
    d = self.datasets[dataset_id][dataset_sample_id]
    
    args = get_args()
    iteration = idx // args.global_batch_size
    
    if iteration < args.train_iters:
        # `iteration` can exceed train_iters due to the prefetching of the next batch
        train_data_dump_path = Path(args.train_data_dump_path) / f"train_data_{args.rank}.jsonl"
        with train_data_dump_path.open("a", encoding="utf-8") as f:
            token_ids = list(map(int, d["tokens"]))
            output_text = self.tokenizer.detokenize(token_ids)
            dataset_name = args.data_path[2 * dataset_id + 1]  # weight, path, weight, path, ...
            row = {
                "iteration": iteration,
                "dataset_idx": int(dataset_id),
                "dataset_name": dataset_name,
                "doc_ids": list(map(lambda n: int(n), d['doc_ids'])),
                "text": output_text,
                "token_ids": token_ids,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del d["doc_ids"]  # Remove the doc_ids from the output as it is not used in the forward pass
    return {"dataset_id": dataset_id, **d}


BlendedDataset.__getitem__ = blended_dataset_getitem  # Monkey-patch the BlendedDataset class


torch.distributed.get_rank = lambda: 0  # Mock the torch.distributed.get_rank function


def gpt_dataset_getitem(self, idx: int) -> Dict[str, torch.Tensor]:
    """Abstract method implementation

    Args:
        idx (int): The index into the dataset

    Returns:
        Dict[str, torch.Tensor]: The text ids wrapped in a dictionary
    """
    text, doc_ids = self._query_document_sample_shuffle_indices(idx)

    text = torch.from_numpy(text).long()
    labels = text[1:].contiguous()
    tokens = text[:-1].contiguous()

    assert not torch.any(
        tokens >= self.config.tokenizer.vocab_size
    ), "An input token is out of bounds of the tokenizer vocabulary"

    if (
        not self.masks_and_position_ids_are_cacheable
        or not self.masks_and_position_ids_are_cached
    ):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )
        if self.masks_and_position_ids_are_cacheable:
            self.cached_attention_mask = attention_mask
            self.cached_loss_mask = loss_mask
            self.cached_position_ids = position_ids
            self.masks_and_position_ids_are_cached = True
    else:
        attention_mask = self.cached_attention_mask
        loss_mask = self.cached_loss_mask
        position_ids = self.cached_position_ids

    if self.config.create_attention_mask:
        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "doc_ids": doc_ids,
        }
    else:
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "doc_ids":doc_ids
        }


GPTDataset.__getitem__ = gpt_dataset_getitem  # Monkey-patch the GPTDataset class


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=_add_train_data_dumping_args,
        allow_no_cuda=True,
        skip_mpu_initialization=True,
    )

    args = get_args()

    args.iteration = 0

    train_ds, _, _ = build_train_valid_test_datasets(
        train_valid_test_datasets_provider
    )
    train_ds.tokenizer = build_tokenizer(args)

    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(train_ds),
        consumed_samples=args.consumed_train_samples,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    train_data_iterator = iter(train_dataloader)

    os.makedirs(args.train_data_dump_path, exist_ok=True)
    
    num_instances_to_train = args.train_iters * args.global_batch_size
    num_seen_instances = 0
    for batch in train_data_iterator:
        num_seen_instances += len(batch["tokens"])
        if num_seen_instances >= num_instances_to_train:
            assert num_seen_instances == num_instances_to_train
            break
