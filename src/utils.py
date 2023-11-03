# deep learning libraries
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

# other libraries
import os
import random
from typing import Literal, Tuple


def load_data(
    dataset_name: Literal["QM9"],
    save_path: str,
    split_sizes: Tuple[float, float, float],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # load dataset
    dataset: InMemoryDataset
    if dataset_name == "QM9":
        # get dataset
        dataset = torch_geometric.datasets.QM9(
            root=save_path,
        )
    else:
        raise ValueError("Invalid dataset name")

    # shuffle and get length
    dataset = dataset.shuffle()
    len_dataset = len(dataset)

    # get datasets
    train_dataset: InMemoryDataset = dataset[: int(split_sizes[0] * len_dataset)]
    val_dataset: InMemoryDataset = dataset[
        int(split_sizes[0] * len_dataset) : int(
            (split_sizes[0] + split_sizes[1]) * len_dataset
        )
    ]
    test_dataset: InMemoryDataset = dataset[
        int((split_sizes[0] + split_sizes[1]) * len_dataset) :
    ]

    # get dataloaders
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=2048)
    val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=2048)
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=2048)

    return train_dataloader, val_dataloader, test_dataloader


@torch.no_grad()
def compute_predictions_by_sampling(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch_indexes: torch.Tensor,
    num_samples,
) -> torch.Tensor:
    # create original tensor for predictions
    predictions: torch.Tensor = torch.zeros(
        num_samples, *model(x, edge_index, batch_indexes).shape
    ).to(x.device)

    # iter over samples to collect predictions
    for sample in range(num_samples):
        predictions[sample] = model(x, edge_index, batch_indexes)

    # copmpute mean of predictions
    predictions = torch.mean(predictions, dim=0)

    return predictions


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: int
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
