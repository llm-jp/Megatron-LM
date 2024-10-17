import torch
from pretrain_gpt import train_valid_test_datasets_provider
from megatron.training.initialize import initialize_megatron
from megatron.training.training import (
    get_args,
    build_train_valid_test_datasets,
)
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler


if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    args = get_args()

    args.iteration = 0
    args.num_floating_point_operations_so_far = 0

    train_ds, _, _ = build_train_valid_test_datasets(
        train_valid_test_datasets_provider
    )

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

    for batch in train_data_iterator:
        print(batch.keys())
        print(batch["dataset_id"])
        print(batch["tokens"].shape)
        break
