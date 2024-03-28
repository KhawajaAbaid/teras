import torch
from torch.utils.data import Dataset


class GAINDataset(Dataset):
    def __init__(self, x, seed: int = 1337):
        self.x = x
        self.num_samples = x.shape[0]
        self.seed = seed
        torch.manual_seed(self.seed)
        self.random_idx = torch.randperm(self.num_samples)

    def __getitem__(self, item):
        return self.x[item], self.x[self.random_idx][item]

    def __len__(self):
        return self.num_samples


def create_gain_dataset(x, seed: int = 1337):
    """
    Creates a torch dataloader compatible with the `GAIN` architecture.
    The resultant dataset produces a tuple consisting of a batch of data for
    `generator` and another batch of data for the `discriminator`.

    Args:
        x: Dataset to use for training.
        seed: int, seed to use in shuffling. Defaults to 1337.
    """
    gain_ds = GAINDataset(x, seed=seed)
    return gain_ds
