import logging
import numpy as np
import torch

from model.Time_encoder import TimeEncoder
from modules.embedding_module import get_embedding_module

class STG(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=1, dropout=0.1,time_dimension=100,
               n_neighbors=None,n_filters=None,
               ):
    super(STG, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.n_filters = n_filters
    self.time_encoder = TimeEncoder(time_dim=time_dimension)


    self.embedding_module = get_embedding_module(module_type="STG",
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=time_dimension,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 dropout=dropout,
                                                 n_neighbors = self.n_neighbors, n_filters = self.n_filters
                                                 )


  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, n_neighbors=20,n_filters=4):
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times])


    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             n_filters = n_filters,
                                                             )

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

    return source_node_embedding, destination_node_embedding


  def compute_edge_probabilities(self, source_nodes, destination_nodes, edge_times, edge_idxs,n_neighbors=20, n_filters = 4):

    src_node_embeddings, dst_node_embeddings= self.compute_temporal_embeddings(source_nodes, destination_nodes,
                                                                                edge_times, edge_idxs, n_neighbors,n_filters)

    return src_node_embeddings, dst_node_embeddings


  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
