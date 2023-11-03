# deep lerning libraries
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# other libraries
import os
from tqdm.auto import tqdm
from typing import Literal, Tuple

# own modules
from src.utils import set_seed
from src.utils import load_data
from src.models import GCN, GAT
from src.metrics import pi_ci

# set seed, number of therads and define device
set_seed(42)
torch.set_num_threads(8)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define static variables
DATA_PATH: str = "data"
SAVE_PATH: str = "models"
RESULTS_PATH: str = "results"
SPLIT_SIZES: Tuple[float, float, float] = (0.7, 0.15, 0.15)


def main() -> None:
    """
    This function is the main program for training models

    Raises:
        ValueError: Invalid model name
    """

    # define variables
    dataset_name: Literal["QM9"] = "QM9"
    model_name: Literal["gcn", "gat"] = "gat"
    bayesian: bool = True

    # define hyperparameters
    lr: float = 1e-3
    num_hidden_layers: int = 2
    kl_weight: float = 1.0
    epochs: int = 100
    num_bayesian_samples = 50

    # check device
    print(f"device: {device}")

    # define bayesian name full
    bayesian_name: str
    if bayesian:
        bayesian_name = f"{bayesian}_{kl_weight}"
    else:
        bayesian_name = str(bayesian)

    # define dataset
    test_data: DataLoader
    _, _, test_data = load_data(
        dataset_name, f"{DATA_PATH}/{dataset_name}", SPLIT_SIZES
    )

    # define model
    data: Data = next(iter(test_data))
    model: torch.nn.Module
    if model_name == "gcn":
        model = GCN(data.x.shape[1], num_hidden_layers, 1, bayesian).to(device)
    elif model_name == "gat":
        model = GAT(data.x.shape[1], num_hidden_layers, 1, bayesian).to(device)
    else:
        raise ValueError("Invalid model_name")

    # define name
    name: str = f"d_{dataset_name}_m_{model_name}_nl_{num_hidden_layers}_lr_{lr}_b_{bayesian_name}"

    # load weights
    model.load_state_dict(torch.load(f"{SAVE_PATH}/{name}/state_dict.pt"))

    # define metrics
    mae: torch.nn.Module = torch.nn.L1Loss()

    model.eval()
    with torch.no_grad():
        # init metrics lists
        maes = []
        wcis1 = []
        wcis2 = []
        wcis3 = []

        # iterate over data
        for data in tqdm(test_data):
            x = data.x.float().to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.float().to(device)
            batch_indexes = data.batch.to(device)

            # compute outputs and loss value
            outputs = model(x, edge_index, batch_indexes)

            predictions: torch.Tensor = torch.zeros(
                num_bayesian_samples, *outputs.shape
            ).to(device)

            for i in range(num_bayesian_samples):
                predictions[i] = model(x, edge_index, batch_indexes)

            # add metrics to metrics lists
            maes.append(
                mae(torch.mean(predictions, dim=0), y[:, 0].unsqueeze(1)).item()
            )
            wcis1.append(pi_ci(predictions, y[:, 0].unsqueeze(1), 1).item())
            wcis2.append(pi_ci(predictions, y[:, 0].unsqueeze(1), 2).item())
            wcis3.append(pi_ci(predictions, y[:, 0].unsqueeze(1), 3).item())

        # save results into dataframe
        df: pd.DataFrame = pd.DataFrame(
            columns=["mae", "within ci std 1", "within ci std 2", "within ci std 3"],
            data=np.array(
                [np.mean(maes), np.mean(wcis1), np.mean(wcis2), np.mean(wcis3)]
            )[np.newaxis, :],
        )

        # create dir if it does not exist
        if not os.path.isdir(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

        # save pandas dataframe
        df.to_csv(f"{RESULTS_PATH}/{name}.csv")


if __name__ == "__main__":
    main()
