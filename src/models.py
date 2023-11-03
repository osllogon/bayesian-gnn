# deep learning libraries
import torch
import torchbnn
import torch_geometric
import torch.nn.functional
from torch_geometric.nn import global_mean_pool

# own imports
from typing import Tuple, List
from src.bayesian_models import BayesianGCNConv, BayesianGATv2Conv


class GCN(torch.nn.Module):
    """
    This class defines a model with two GCN layers

    Attributes:
        conv1: first GNN
        relu: non linearity function
        dropout: dropout layer
        conv2: second GNN
    """

    def __init__(
        self,
        input_size: int,
        num_hidden_layers: int,
        output_size: int,
        bayesian: bool,
    ) -> None:
        """
        This method is the constructor for the GCN class

        Args:
            input_size: number of node features
            hidden_channels: size between the layers
            output_size: number of possible classes
        """

        # call superclass constructor
        super().__init__()

        # append hidden layers
        layers: List[torch_geometric.nn.MessagePassing] = []
        complete_sizes: Tuple[int, ...] = (input_size,) + tuple(
            [64 for _ in range(num_hidden_layers)]
        )
        for i in range(len(complete_sizes) - 1):
            if bayesian:
                layers.append(BayesianGCNConv(complete_sizes[i], complete_sizes[i + 1]))
            else:
                layers.append(
                    torch_geometric.nn.GCNConv(complete_sizes[i], complete_sizes[i + 1])
                )

        # get gnn model
        self.gnn: torch.nn.Module = torch_geometric.nn.Sequential(
            "x, edge_index", [(layer, "x, edge_index -> x") for layer in layers]
        )

        # define lin
        self.lin: torch.nn.Module
        if bayesian:
            self.lin = torchbnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=64,
                out_features=output_size,
                bias=True,
            )
        else:
            self.lin = torch.nn.Linear(64, output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method defines the forward pass

        Args:
            inputs: node matrix. Dimensions: [batch size, input size]
            edge_index: edge index tensor that represents the adj matrix. Dimensions: [2, number of edges]

        Returns:
            predictions of the classes. Dimensions: [batch size, output size]
        """

        # compute gnn outputs
        outputs: torch.Tensor = self.gnn(inputs, edge_index)

        # compute GAP
        outputs = global_mean_pool(outputs, batch_indexes)

        # compute final prediction
        outputs = self.lin(outputs)

        return outputs


class GAT(torch.nn.Module):
    """
    This class defines a model with two GCN layers

    Attributes:
        conv1: first GNN
        relu: non linearity function
        dropout: dropout layer
        conv2: second GNN
    """

    def __init__(
        self,
        input_size: int,
        num_hidden_layers: int,
        output_size: int,
        bayesian: bool,
    ) -> None:
        """
        This method is the constructor for the GCN class

        Args:
            input_size: number of node features
            hidden_channels: size between the layers
            output_size: number of possible classes
        """

        # call superclass constructor
        super().__init__()

        # append hidden layers
        layers: List[torch_geometric.nn.MessagePassing] = []
        complete_sizes: Tuple[int, ...] = (input_size,) + tuple(
            [64 for _ in range(num_hidden_layers)]
        )
        for i in range(len(complete_sizes) - 1):
            if bayesian:
                layers.append(
                    BayesianGATv2Conv(complete_sizes[i], complete_sizes[i + 1])
                )
            else:
                layers.append(
                    torch_geometric.nn.GATv2Conv(
                        complete_sizes[i], complete_sizes[i + 1]
                    )
                )

        # get gnn model
        self.gnn: torch.nn.Module = torch_geometric.nn.Sequential(
            "x, edge_index", [(layer, "x, edge_index -> x") for layer in layers]
        )

        # define lin
        self.lin: torch.nn.Module
        if bayesian:
            self.lin = torchbnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=64,
                out_features=output_size,
                bias=True,
            )
        else:
            self.lin = torch.nn.Linear(64, output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method defines the forward pass

        Args:
            inputs: node matrix. Dimensions: [batch size, input size]
            edge_index: edge index tensor that represents the adj matrix. Dimensions: [2, number of edges]

        Returns:
            predictions of the classes. Dimensions: [batch size, output size]
        """

        # compute gnn outputs
        outputs: torch.Tensor = self.gnn(inputs, edge_index)

        # compute GAP
        outputs = global_mean_pool(outputs, batch_indexes)

        # compute final prediction
        outputs = self.lin(outputs)

        return outputs


# class GAT(torch.nn.Module):
#     """
#     This class defines a model with two GCN layers

#     Attributes:
#         conv1: first GNN
#         relu: non linearity function
#         dropout: dropout layer
#         conv2: second GNN
#     """

#     def __init__(
#         self,
#         input_size: int,
#         hidden_sizes: Tuple[int, ...],
#         output_size: int,
#         bayesian: bool,
#     ) -> None:
#         """
#         This method is the constructor for the GCN class

#         Args:
#             input_size: number of node features
#             hidden_channels: size between the layers
#             output_size: number of possible classes
#         """

#         # call superclass constructor
#         super().__init__()

#         # append hidden layers
#         layers: List[torch_geometric.nn.MessagePassing] = []
#         complete_sizes: Tuple[int, ...] = (input_size,) + hidden_sizes
#         for i in range(len(complete_sizes) - 1):
#             if bayesian:
#                 layers.append(BayesianGCNConv(complete_sizes[i], complete_sizes[i + 1]))
#             else:
#                 layers.append(
#                     torch_geometric.nn.GATv2Conv(complete_sizes[i], complete_sizes[i + 1])
#                 )

#         # get gnn model
#         self.gnn: torch.nn.Module = torch.nn.Sequential(*layers)

#         # define lin
#         self.lin: torch.nn.Module
#         if bayesian:
#             self.lin = torchbnn.BayesLinear(
#                 prior_mu=0,
#                 prior_sigma=0.1,
#                 in_features=hidden_size,
#                 out_features=output_size,
#                 bias=True,
#             )
#         else:
#             self.lin = torch.nn.Linear(hidden_size, output_size)

#     def forward(
#         self,
#         inputs: torch.Tensor,
#         edge_index: torch.Tensor,
#         batch_indexes: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         This method defines the forward pass

#         Args:
#             inputs: node matrix. Dimensions: [batch size, input size]
#             edge_index: edge index tensor that represents the adj matrix. Dimensions: [2, number of edges]

#         Returns:
#             predictions of the classes. Dimensions: [batch size, output size]
#         """

#         # compute gnn outputs
#         outputs: torch.Tensor = self.gnn(inputs, edge_index)

#         # compute GAP
#         outputs = global_mean_pool(outputs, batch_indexes)

#         # compute final prediction
#         outputs = self.lin(outputs)

#         return outputs
