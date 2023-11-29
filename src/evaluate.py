# deep lerning libraries
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# other libraries
import os
from tqdm.auto import tqdm
from typing import Literal, Tuple, List, Optional

# own modules
from src.utils import set_seed
from src.utils import load_data
from src.models import GCN, GAT
from src.metrics import percentage_inside, distance_distributions

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
    bayesian_mode: Literal["none", "weights", "dropout"] = "none"
    dropout_rate: Optional[float] = None

    # define hyperparameters
    lr: float = 1e-3
    num_hidden_layers: int = 2
    kl_weight: float = 1.0
    epochs: int = 100
    num_bayesian_samples: int = 50

    # check device
    print(f"device: {device}")

    # define bayesian name full
    bayesian_name: str
    if bayesian_mode == "none":
        bayesian_name = str(bayesian_mode)

    elif bayesian_mode == "weights":
        bayesian_name = f"{bayesian_mode}_{kl_weight}"

    elif bayesian_mode == "dropout":
        bayesian_name = f"{bayesian_mode}_{dropout_rate}"

    else:
        raise ValueError("Invalid bayesian_mode")

    # define dataset
    test_data: DataLoader
    train_data, _, test_data = load_data(
        dataset_name, f"{DATA_PATH}/{dataset_name}", SPLIT_SIZES
    )

    # define model
    data: Data = next(iter(test_data))
    model: torch.nn.Module
    if model_name == "gcn":
        model = GCN(
            data.x.shape[1], num_hidden_layers, 1, bayesian_mode, dropout_rate
        ).to(device)
    elif model_name == "gat":
        model = GAT(
            data.x.shape[1], num_hidden_layers, 1, bayesian_mode, dropout_rate
        ).to(device)
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
        maes: List[float] = []
        normal_tests: List[float] = []
        percentages_insides: Tuple[List[float], List[float], List[float]] = ([], [], [])
        distances_distributions: Tuple[List[float], List[float], List[float]] = (
            [],
            [],
            [],
        )

        # iterate over data
        for data in tqdm(test_data):
            x: torch.Tensor = data.x.float().to(device)
            edge_index: torch.Tensor = data.edge_index.to(device)
            y: torch.Tensor = data.y.float().to(device)
            batch_indexes: torch.Tensor = data.batch.to(device)

            # compute outputs to have the shape
            outputs: torch.Tensor = model(x, edge_index, batch_indexes)

            # compute predictions
            predictions: torch.Tensor = torch.zeros(
                num_bayesian_samples, *outputs.shape
            ).to(device)

            for i in range(num_bayesian_samples):
                predictions[i] = model(x, edge_index, batch_indexes)

            # add metrics to metrics lists
            normal_tests.append(
                stats.normaltest(predictions.detach().cpu().numpy(), axis=0)[1].mean()
            )
            maes.append(
                mae(torch.mean(predictions, dim=0), y[:, 0].unsqueeze(1)).item()
            )

            for i in range(len(percentages_insides)):
                percentages_insides[i].append(
                    percentage_inside(predictions, y[:, 0].unsqueeze(1), i + 1).item()
                )
                distances_distributions[i].append(percentages_insides[i][-1])

        # save results into dataframe
        df: pd.DataFrame = pd.DataFrame(
            columns=[
                "mae",
                "normal test",
                "percentage inside ci std 1",
                "percentage inside ci std 2",
                "percentage inside ci std 3",
                "distance distributions ci std 1",
                "distance distributions ci std 2",
                "distance distributions ci std 3",
            ],
            data=np.array(
                [
                    np.mean(maes),
                    np.mean(normal_tests),
                    np.mean(percentages_insides[0]),
                    np.mean(percentages_insides[1]),
                    np.mean(percentages_insides[2]),
                    np.mean(distances_distributions[0]),
                    np.mean(distances_distributions[1]),
                    np.mean(distances_distributions[2]),
                ]
            )[np.newaxis, :],
        )

        # create dir if it does not exist
        if not os.path.isdir(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

        # save pandas dataframe
        df.to_csv(f"{RESULTS_PATH}/{name}.csv")


if __name__ == "__main__":
    main()
