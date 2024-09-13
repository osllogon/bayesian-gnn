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
from typing import Literal, Optional

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
SPLIT_SIZES: tuple[float, float, float] = (0.7, 0.15, 0.15)


def main() -> None:
    """
    This function is the main program for training models

    Raises:
        ValueError: Invalid model name
    """

    # define variables
    dataset_name: Literal["QM9", "ZINC"] = "QM9"
    model_name: Literal["gcn", "gat"] = "gat"
    bayesian_mode: Literal["none", "weights", "dropout"] = "weights"
    dropout_rate: Optional[float] = 0.5

    # define hyperparameters
    lr: float = 1e-3
    num_hidden_layers: int = 2
    kl_weight: float = 1.0
    epochs: int = 100
    num_bayesian_samples: int = 500

    # empty nohup file
    open("nohup.out", "w").close()

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
    train_data, _, test_data, x_scaler, y_scaler = load_data(
        dataset_name, f"{DATA_PATH}/{dataset_name}", SPLIT_SIZES
    )

    # change device of scalers
    x_scaler = x_scaler.to(device)
    y_scaler = y_scaler.to(device)

    # define model
    data: Data = next(iter(test_data))
    model: torch.nn.Module
    if model_name == "gcn":
        model = GCN(
            data.x.shape[1],
            num_hidden_layers,
            data.y.shape[1],
            bayesian_mode,
            dropout_rate,
        ).to(device)
    elif model_name == "gat":
        model = GAT(
            data.x.shape[1],
            num_hidden_layers,
            data.y.shape[1],
            bayesian_mode,
            dropout_rate,
        ).to(device)
    else:
        raise ValueError("Invalid model_name")

    # define name
    name: str = f"d_{dataset_name}_m_{model_name}_nl_{num_hidden_layers}_lr_{lr}_b_{bayesian_name}"

    # load weights
    model.load_state_dict(torch.load(f"{SAVE_PATH}/{name}/state_dict.pt"))

    # define metrics
    mae: torch.nn.Module = torch.nn.L1Loss()

    # define metrics names
    metrics_names: list[str] = [
        "MAE",
        "pi-ci std 1",
        "pi-ci std 2",
        "pi-ci std 3",
        "pi-ci std 0.1",
        "pi-ci std 0.2",
        "pi-ci std 0.3",
    ]

    # create metrics
    metrics: np.ndarray = np.zeros((data.y.shape[1], len(metrics_names)))

    model.train()
    with torch.no_grad():
        # init metrics lists
        maes: list[float] = []
        normal_tests: list[float] = []
        percentages_insides: tuple[list[float], list[float], list[float]] = ([], [], [])
        distances_distributions: tuple[list[float], list[float], list[float]] = (
            [],
            [],
            [],
        )

        # iterate over data
        for data in tqdm(test_data):
            x: torch.Tensor = x_scaler.transform(data.x.float().to(device))
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
                predictions[i] = y_scaler.inverse_transform(
                    model(x, edge_index, batch_indexes)
                )

            for j in range(data.y.shape[1]):
                # add metrics to metrics lists
                # metrics[j, 0] += stats.normaltest(
                #     predictions[:, j].detach().cpu().numpy(), axis=0
                # )[1].mean()
                metrics[j, 0] += mae(
                    torch.mean(predictions[:, :, j], dim=0), y[:, j]
                ).item()

                for i in range(len(percentages_insides)):
                    percentage_inside_value = percentage_inside(
                        predictions[:, :, j], y[:, j], i + 1
                    ).item()
                    metrics[j, 1 + i] += percentage_inside_value
                    metrics[j, 4 + i] += distance_distributions(
                        percentage_inside_value, i + 1
                    )

        # divide to compute average
        metrics /= len(test_data)

        # save results into dataframe
        df: pd.DataFrame = pd.DataFrame(
            columns=metrics_names,
            data=metrics,
        )

        # create dir if it does not exist
        if not os.path.isdir(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

        # save pandas dataframe
        df.to_csv(f"{RESULTS_PATH}/{name}.csv")
        df.to_latex(
            open(f"{RESULTS_PATH}/{name}.tex", "w"), index=True, float_format="%.3f"
        )


if __name__ == "__main__":
    main()
