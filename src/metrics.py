# deep learning libraries
import torch
import numpy as np


@torch.no_grad()
def percentage_inside(
    predictions: torch.Tensor, target: torch.Tensor, num_stds: int
) -> torch.Tensor:
    """
    This fucntion computes the percentage of samples inside confidence 
    intervals

    Args:
        predictions: predictions tensor. Dimensions:
            [number of samples, number of predictions, 1]
        target: target tensor. Dimenions: [number of nodes, 1]
        num_stds: number of standard deviations

    Returns:
        percentage of samples inside the confidence interval
    """

    # define limits of confidence intervals
    upper_ci = torch.mean(predictions, dim=0) + torch.quantile(
        predictions, num_stds * torch.tensor([0.25]).to(predictions.device), dim=0
    )
    torch.std(predictions, dim=0)
    low_ci = torch.mean(predictions, dim=0) - torch.quantile(
        predictions, num_stds * torch.tensor([0.25]).to(predictions.device), dim=0
    )

    # count inside samples
    inside_samples: torch.Tensor = (target <= upper_ci) & (target >= low_ci)
    num_inside_samples: torch.Tensor = torch.sum(inside_samples) / predictions.shape[1]

    return num_inside_samples


@torch.no_grad()
def distance_distributions(percentage: float, num_stds: int) -> float:
    """
    This function computes the distance between the percentage of
    samples inside the confidence interval and percentage of samples
    inside 1 std of a gaussian distribution

    Args:
        percentage: percentage of samples inside the confidence
            interval
        num_stds: number of standard deviations

    Raises:
        ValueError: Invalid number of stds, only 1 to 3

    Returns:
        distance between the percentage of samples inside the
            confidence interval and percentage of samples inside 1 std
            of a gaussian distribution
    """

    # compute distance from the percentage to real confidence interval
    distance: float
    if num_stds == 1:
        distance = abs(percentage - 0.68)

    elif num_stds == 2:
        distance = abs(percentage - 0.95)

    elif num_stds == 3:
        distance = abs(percentage - 0.997)

    else:
        raise ValueError("Invalid number of stds, only 1 to 3")

    return distance
