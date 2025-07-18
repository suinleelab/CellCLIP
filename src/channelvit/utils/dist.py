"""Utility finction"""
import torch.distributed as dist


def is_dist_avail_and_initialized():
    """Utility function"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Utility function"""

    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
