# deep lerning libraries
import torch
import torchbnn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# other libraries
import os
from tqdm.auto import tqdm
from typing import Literal, Tuple, Optional

# own modules
from src.utils import set_seed, load_data, compute_predictions_by_sampling
from src.models import GCN, GAT

# set seed, number of therads and define device
set_seed(42)
torch.set_num_threads(8)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define static variables
DATA_PATH: str = "data"
SAVE_PATH: str = "models"
SPLIT_SIZES: Tuple[float, float, float] = (0.7, 0.15, 0.15)


def main() -> None:
    """
    This function is the main program for training models

    Raises:
        ValueError: Invalid model name
    """

    # define variables
    dataset_name: Literal["QM9", "ZINC"] = "QM9"
    model_name: Literal["gcn", "gat"] = "gcn"
    bayesian_mode: Literal["none", "weights", "dropout"] = "weights"
    dropout_rate: Optional[float] = 0.5

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

    # define name and tensorboard writer
    name: str = f"d_{dataset_name}_m_{model_name}_nl_{num_hidden_layers}_lr_{lr}_b_{bayesian_name}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define dataset
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(
        dataset_name, f"{DATA_PATH}/{dataset_name}", SPLIT_SIZES
    )

    # define model
    data: Data = next(iter(train_data))
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

    # define losses
    loss: torch.nn.Module = torch.nn.MSELoss()
    kl_loss: torch.nn.Module = torchbnn.BKLLoss()

    # define optimizer
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # define metrics
    mae: torch.nn.Module = torch.nn.L1Loss()

    # iter over epochs
    epoch: int
    for epoch in tqdm(range(epochs)):
        # activate train mode
        model.train()

        # init vectors
        loss_totals = []
        loss_mses = []
        loss_kls = []
        maes: list[list[float]] = [[] for _ in range(data.y.shape[1])]

        # iterate over data
        for data in train_data:
            x = data.x.float().to(device)
            edge_index = data.edge_index.to(device)
            y: torch.Tensor = data.y.float().to(device)
            batch_indexes = data.batch.to(device)

            # compute outputs and loss value
            outputs: torch.Tensor = model(x, edge_index, batch_indexes)
            loss_mse = loss(outputs, y)
            loss_kl = kl_weight / len(train_data) * kl_loss(model)
            loss_value = loss_mse + loss_kl

            # optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # add data
            loss_totals.append(loss_value.item())
            loss_mses.append(loss_mse.item())
            loss_kls.append(loss_kl.item())
            for i in range(data.y.shape[1]):
                maes[i].append(mae(outputs[:, i], y[:, i]).item())

        # writer on tensorboard
        writer.add_scalar("loss_total/train", np.mean(loss_totals), epoch)
        writer.add_scalar("loss_mse/train", np.mean(loss_mses), epoch)
        writer.add_scalar("loss_kl/train", np.mean(loss_kls), epoch)
        for i in range(data.y.shape[1]):
            writer.add_scalar(f"mae/{i}/train", np.mean(maes[i]), epoch)

        # activate eval mode
        model.eval()

        # decativate the gradient
        with torch.no_grad():
            # init vectors
            maes = [[] for _ in range(data.y.shape[1])]

            # iterate over data
            for data in val_data:
                x = data.x.float().to(device)
                edge_index = data.edge_index.to(device)
                y = data.y.float().to(device)
                batch_indexes = data.batch.to(device)

                # compute outputs and loss value
                outputs = model(x, edge_index, batch_indexes)
                if bayesian_mode != "none":
                    outputs = compute_predictions_by_sampling(
                        model, x, edge_index, batch_indexes, num_bayesian_samples
                    )
                else:
                    outputs = model(x, edge_index, batch_indexes)

                # add data
                for i in range(data.y.shape[1]):
                    maes[i].append(mae(outputs[:, i], y[:, i]).item())

            # writer on tensorboard
            for i in range(data.y.shape[1]):
                writer.add_scalar(f"mae/{i}/val", np.mean(maes[i]), epoch)

    # create dirs to save model
    if not os.path.exists(f"{SAVE_PATH}/{name}"):
        os.makedirs(f"{SAVE_PATH}/{name}")

    # save model
    model = model.cpu()
    torch.save(model.state_dict(), f"{SAVE_PATH}/{name}/state_dict.pt")


if __name__ == "__main__":
    main()
