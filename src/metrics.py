# deep learning libraries
import torch


def pi_ci(
    predictions: torch.Tensor, target: torch.Tensor, num_stds: int
) -> torch.Tensor:
    # define limits of ci
    upper_ci = torch.mean(predictions, dim=0) + num_stds * torch.std(predictions, dim=0)
    low_ci = torch.mean(predictions, dim=0) - num_stds * torch.std(predictions, dim=0)

    # coutn inside samples
    inside_samples: torch.Tensor = (target <= upper_ci) & (target >= low_ci)
    num_inside_samples: torch.Tensor = torch.sum(inside_samples) / predictions.shape[1]

    return num_inside_samples
