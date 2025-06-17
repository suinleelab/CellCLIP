"""Modified graphium class that enables latent extraction before predictor head"""

import torch
from graphium.nn.architectures.global_architectures import FullGraphMultiTaskNetwork
from torch_geometric.data import Batch


class FullGraphMultiTaskNetworkNew(FullGraphMultiTaskNetwork):
    """Modified class that enable latent extraction before predictor head"""

    def forward(self, g: Batch) -> torch.tensor:
        """Modified forward function to return latents from gnn"""

        # Apply the positional encoders
        g = self.encoder_manager(g)

        e = None

        # Run the pre-processing network on node features
        if self.pre_nn is not None:
            g["feat"] = self.pre_nn.forward(g["feat"])

        # Run the pre-processing network on edge features
        # If there are no edges, skip the forward and change the dimension of e
        if self.pre_nn_edges is not None:
            e = g["edge_feat"]
            if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                e = torch.zeros(
                    list(e.shape[:-1]) + [self.pre_nn_edges.out_dim],
                    device=e.device,
                    dtype=e.dtype,
                )
            else:
                e = self.pre_nn_edges.forward(e)
            g["edge_feat"] = e

        # Run the graph neural network
        # passing graph features to task head for feature extraction.

        g = self.gnn.forward(g)

        features = {
            task_level: self.task_heads.graph_output_nn[task_level](g)
            for task_level in self.task_heads.task_levels
        }

        return features["graph"]
