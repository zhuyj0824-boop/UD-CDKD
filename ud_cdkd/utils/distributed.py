from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialise_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = world_size > 1
    if enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        return DistributedContext(True, rank, local_rank, dist.get_world_size(), device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(False, 0, 0, 1, device)


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        dist.barrier()


def all_reduce_sum(value: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if ctx.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()
