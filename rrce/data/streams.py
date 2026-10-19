import numpy as np
from torch.utils.data import DataLoader


class BalancedConceptStream:
    def __init__(self, concepts):
        self.concepts = list(concepts)
        self.order = []
        self.cursor = 0
        if not self.concepts:
            raise ValueError("Concept stream cannot be empty")

    def next(self):
        if self.cursor == len(self.order):
            self.order = list(self.concepts)
            np.random.shuffle(self.order)
            self.cursor = 0
        value = self.order[self.cursor]
        self.cursor += 1
        return value


def make_loader(dataset, batch_size, runtime):
    options = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.pin_memory and runtime.device.startswith("cuda"),
    }
    if runtime.num_workers:
        options.update(
            prefetch_factor=runtime.prefetch_factor, persistent_workers=runtime.persistent_workers
        )
    return DataLoader(dataset, **options)


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
