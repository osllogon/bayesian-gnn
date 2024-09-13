# deep learning libraries
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

# other libraries
import os
import random
from typing import Literal, Tuple


class MinMaxScaler(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        """
        This method is the constructor of MinMaxScaler class

        Args:
            dimension of scalers
        """

        # call superclass constructor
        super().__init__()

        self.max: torch.Tensor = torch.nn.Parameter(torch.zeros(1, dim))
        self.min: torch.Tensor = torch.nn.Parameter(torch.zeros(1, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward call

        Args:
            input tensor. Dimensions: [rows, columns]

        Returns:
            output tensor. Dimensions: [rows, columns]
        """

        outputs: torch.Tensor = (inputs - self.min) / (self.max - self.min + 1e-6)

        outputs = torch.round(outputs, decimals=5)

        return outputs

    def fit(self, dataloader: DataLoader, arg_name: str) -> None:
        """
        This function trains the scaler

        Args:
            input tensor. Dimensions: [rows, columns]
        """

        batch: Data = next(iter(dataloader))

        min_dataset: torch.Tensor = torch.amin(batch[arg_name], dim=0, keepdim=True)
        max_dataset: torch.Tensor = torch.amax(batch[arg_name], dim=0, keepdim=True)

        # compute the max and min based on the inputs
        for batch in dataloader:
            # compute batch metrics
            min_batch: torch.Tensor = torch.amin(batch[arg_name], dim=0, keepdim=True)
            max_batch: torch.Tensor = torch.amax(batch[arg_name], dim=0, keepdim=True)

            # update dataset values
            min_dataset[min_batch < min_dataset] = min_batch[min_batch < min_dataset]
            max_dataset[max_batch > max_dataset] = max_batch[max_batch > max_dataset]

        self.min = torch.nn.Parameter(min_dataset)
        self.max = torch.nn.Parameter(max_dataset)

        return None

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function normalizes the input data

        Args:
            input tensor. Dimensions: [rows, columns]

        Returns:
            output tensor. Dimensions: [rows, columns]
        """

        # transform inputs
        outputs: torch.Tensor = self.forward(inputs)

        return outputs

    def inverse_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function unnormalizes the input data

        Args:
            input tensor. Dimensions: [rows, columns]

        Returns:
            output tensor. Dimensions: [rows, columns]
        """

        # compute inverse transform
        outputs: torch.Tensor = inputs * (self.max - self.min) + self.min

        outputs = torch.round(outputs, decimals=5)

        return outputs


def load_data(
    dataset_name: Literal["QM9", "ZINC"],
    save_path: str,
    split_sizes: Tuple[float, float, float],
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:
    """
    This function loads the data.

    Args:
        dataset_name: name of the dataset.
        save_path: _description_
        split_sizes: _description_

    Raises:
        ValueError: _description_

    Returns:
        _description_
    """

    # load dataset
    dataset: InMemoryDataset
    if dataset_name == "QM9":
        # get dataset
        dataset = torch_geometric.datasets.QM9(root=save_path)

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

    elif dataset_name == "ZINC":
        # get datasets
        train_dataset = torch_geometric.datasets.ZINC(
            root=f"{save_path}/train", split="train"
        )
        val_dataset = torch_geometric.datasets.ZINC(
            root=f"{save_path}/val", split="val"
        )
        test_dataset = torch_geometric.datasets.ZINC(
            root=f"{save_path}/test", split="test"
        )

    else:
        raise ValueError("Invalid dataset name")

    # get dataloaders
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=2048)
    val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=2048)
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=2048)

    # define scalers
    x_scaler: MinMaxScaler = MinMaxScaler(train_dataset.x.shape[1])
    y_scaler: MinMaxScaler = MinMaxScaler(dataset[0].y.shape[1])

    # fit scalers
    x_scaler.fit(train_dataloader, "x")
    y_scaler.fit(train_dataloader, "y")

    return train_dataloader, val_dataloader, test_dataloader, x_scaler, y_scaler


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
