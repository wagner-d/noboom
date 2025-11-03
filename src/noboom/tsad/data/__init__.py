from .dataset import Dataset

try:
    import torch
except ImportError:
    pass
else:
    from .torch_dataset import TorchDataset