import torch


def norm(x, ord, axis, keepdims):
    return torch.linalg.norm(x, ord=ord, axis=axis, keepdim=keepdims)
