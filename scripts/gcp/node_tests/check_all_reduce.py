import torch
import torch.distributed as torch_distributed
import os
import socket


def setup():
    global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', 1))

    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch_distributed.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    # print(f"rank={global_rank}, local_rank={local_rank}, world_size={world_size} is initialized.")


def cleanup():
    torch_distributed.destroy_process_group()


def check_all_reduce(rank, world_size):
    # initialize a tensor on GPU
    tensor = torch.rand(1).to(rank % torch.cuda.device_count())
    # print(f"Rank {rank} starts with tensor {tensor.item()}")

    # sum all the tensors from different processes
    torch_distributed.all_reduce(
        tensor,
        op=torch_distributed.ReduceOp.SUM
    )
    flag = True
    hostname = socket.gethostname()

    # check if the tensor is the expected sum
    expected_sum = world_size * 0.5  # 0.5 is the expected mean of the random tensor
    local_rank = torch_distributed.get_rank() % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    if not torch.isclose(
        tensor, torch.tensor([expected_sum], device=device), atol=0.5 * world_size
    ):
        print(f"Rank {rank}: hostname={hostname} Check failed! Tensor after all_reduce: {tensor.item()}")
        flag = False

    world_size = int(os.getenv('WORLD_SIZE', '1'))
    world_size_tensor = torch.tensor([world_size], device=device)
    torch_distributed.all_reduce(
        world_size_tensor,
        op=torch_distributed.ReduceOp.MAX
    )
    if world_size_tensor.item() != world_size:
        print(f"Rank {rank}: hostname={hostname} Check failed! world_size after all_reduce: {world_size_tensor.item()}")
        flag = False

    iteration = 18500
    iters_cuda = torch.tensor(
        [iteration],
        dtype=torch.long,
        device=device
    )
    torch_distributed.all_reduce(iters_cuda, op=torch_distributed.ReduceOp.MAX)
    max_iter = iters_cuda[0].item()
    if iteration != max_iter:
        print(f"Rank {rank}: hostname={hostname} Check failed! iteration after all_reduce: {max_iter}")
        flag = False
    iters_cuda = torch.tensor(
        [iteration],
        dtype=torch.long,
        device=device
    )
    torch_distributed.all_reduce(iters_cuda, op=torch_distributed.ReduceOp.MIN)
    min_iter = iters_cuda[0].item()
    if iteration != min_iter:
        print(f"Rank {rank}: hostname={hostname} Check failed! iteration after all_reduce: {min_iter}")
        flag = False

    if flag:
        print(f"Rank {rank}: hostname={hostname} All Check passed!")


def main():
    setup()

    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    check_all_reduce(rank, world_size)
    cleanup()


if __name__ == "__main__":
    main()
