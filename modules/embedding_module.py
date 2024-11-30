import torch
from torch import nn
import numpy as np
torch.set_printoptions(profile="full")

class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, source_nodes, timestamps, n_layers, n_neighbors=None,n_filters=None):
    return NotImplemented

class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout=0.1, n_neighbors=None, n_filters=None):
    super(GraphEmbedding, self).__init__(node_features, edge_features,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.device = device

  def compute_embedding(self, source_nodes, timestamps, n_layers, n_neighbors=20, n_filters=4):

    assert (n_layers >= 0)
    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)

    source_node_features = self.node_features[source_nodes_torch, :]

    if n_layers == 0:
      return source_node_features
    else:
      neighbors_ori, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors_ori.flatten()

      neighbor_embeddings = self.compute_embedding(neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors,
                                                   n_filters = n_filters
                                                   )

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1

      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      source_embedding = self.aggregate(neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        neighbors_ori,edge_times)

      return source_embedding


class STG(GraphEmbedding):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout=0.1,n_neighbors=None,n_filters=None):
      super(STG, self).__init__(node_features, edge_features,
                                neighbor_finder, time_encoder, n_layers,
                                n_node_features, n_edge_features,
                                n_time_features,
                                embedding_dimension, device,
                                dropout, n_neighbors, n_filters)

      self.SIP_encoder = nn.Sequential(
          nn.Linear(in_features=2, out_features=self.n_node_features),
          nn.ReLU(),
          nn.Linear(in_features=self.n_node_features, out_features=self.n_node_features))

      self.MIP_encoder = nn.Sequential(
          nn.Linear(in_features=2, out_features=self.n_node_features),
          nn.ReLU(),
          nn.Linear(in_features=self.n_node_features, out_features=self.n_node_features))

      self.projection_layer_edgetime = nn.Linear(n_edge_features + n_time_features + self.n_node_features,
                                                 self.n_node_features)
      self.complex_weight = nn.Parameter(
          torch.randn(1, n_neighbors // 2 + 1, self.n_node_features, n_filters, 2, dtype=torch.float32) * 0.02)
      self.filter = FilterLayer(max_input_length=n_neighbors, hidden_dim=self.n_node_features, filter_num=n_filters,
                                weight=self.complex_weight)

      self.pooler = Pooler(embed_dim=self.n_node_features)

      self.mlp_mixer = MLPMixer(num_tokens=n_neighbors, num_channels=self.n_node_features)


  def SIPextractor(self, neighbors_tensor: torch.Tensor, edge_times_tensor: torch.Tensor, unique_elements: torch.Tensor,
                   unique_inverse: torch.Tensor, all_edge_diff_ave: torch.Tensor):

      masks = (neighbors_tensor[:, :, None] == unique_elements)
      counts_sip = masks.sum(dim=1, keepdim=True)

      first_times = torch.where(masks, edge_times_tensor[:, :, None], torch.tensor(1e10, device=self.device)).min(dim=1,
                                                                                                                  keepdim=True)[0]
      last_times = \
      torch.where(masks, edge_times_tensor[:, :, None], torch.tensor(0, device=self.device)).max(dim=1, keepdim=True)[0]

      first_times[first_times > torch.max(edge_times_tensor)] = 0

      time_diffs_sip = last_times - first_times

      count_sip_mask = torch.sum(masks * counts_sip, dim=-1)

      time_diffs_sip_mask = torch.sum(masks * time_diffs_sip, dim=-1)
      delta_count_time = (time_diffs_sip_mask / count_sip_mask) / (all_edge_diff_ave + 1e-8)
      density_sip_mask = torch.exp(-1 * torch.abs(1 - delta_count_time))

      count_sip_float = count_sip_mask.float()
      density_sip_float = density_sip_mask.float()

      return count_sip_float, density_sip_float

  def MIPextractor(self, neighbors_tensor: torch.Tensor, edge_times_tensor: torch.Tensor, unique_elements: torch.Tensor,
                   unique_inverse: torch.Tensor, all_edge_diff_ave: torch.Tensor):

      edge_times_flatten = edge_times_tensor.flatten()

      counts_mip = torch.bincount(unique_inverse)

      max_times = torch.full_like(unique_elements, -float('inf'), dtype=torch.float32)
      min_times = torch.full_like(unique_elements, float('inf'))

      max_times.scatter_reduce_(0, unique_inverse, edge_times_flatten, reduce='amax')
      min_times.scatter_reduce_(0, unique_inverse, edge_times_flatten, reduce='amin')

      result_time_diff = max_times - min_times
      result_time_diff_ave = result_time_diff / counts_mip

      delta_count_time = result_time_diff_ave[unique_inverse] / (all_edge_diff_ave + 1e-8)
      density_mip_mask = torch.exp(-1 * torch.abs(1 - delta_count_time))

      count_mip_mask = counts_mip[unique_inverse]

      count_mip_mask = count_mip_mask.reshape(neighbors_tensor.shape[0], neighbors_tensor.shape[1])
      density_mip_mask = density_mip_mask.reshape(neighbors_tensor.shape[0], neighbors_tensor.shape[1])

      count_mip_float = count_mip_mask.float()
      density_mip_float = density_mip_mask.float()

      return count_mip_float, density_mip_float

  def aggregate(self, neighbor_embeddings, edge_time_embeddings, edge_features, neighbors, edge_times):

      neighbors_tensor = torch.from_numpy(neighbors).float().to(self.device)
      edge_times_tensor = torch.from_numpy(edge_times).float().to(self.device)

      neighbors_flatten = neighbors_tensor.flatten()

      unique_elements, unique_inverse = torch.unique(neighbors_flatten, return_inverse=True)

      non_zero_edge_times = edge_times_tensor[edge_times_tensor != 0]
      if non_zero_edge_times.numel() == 0:
          all_edge_diff_ave = 0
      else:
          all_edge_diff_ave = (torch.max(non_zero_edge_times) - torch.min(
              non_zero_edge_times)) / non_zero_edge_times.numel()

      count_sip_float, density_sip_float = self.SIPextractor(neighbors_tensor, edge_times_tensor, unique_elements,
                                                             unique_inverse, all_edge_diff_ave)
      count_mip_float, density_mip_float = self.MIPextractor(neighbors_tensor, edge_times_tensor, unique_elements,
                                                             unique_inverse, all_edge_diff_ave)

      SIP_behavior = torch.stack((count_sip_float, density_sip_float), dim=-1)
      MIP_behavior = torch.stack((count_mip_float, density_mip_float), dim=-1)

      neighbors_tensor[neighbors_tensor > 0] = 1.0

      neighbors_mask = neighbors_tensor.unsqueeze(-1)

      SIP_encoding = self.SIP_encoder(SIP_behavior)
      SIP_encoding = SIP_encoding * neighbors_mask

      MIP_encoding = self.MIP_encoder(MIP_behavior)
      MIP_encoding = MIP_encoding * neighbors_mask

      edge_time_embeddings = edge_time_embeddings * neighbors_mask
      temporal_feat = torch.cat([edge_features, edge_time_embeddings, neighbor_embeddings], dim=-1)
      temporal_feat = self.projection_layer_edgetime(temporal_feat)

      temporal_feat_s = temporal_feat[:np.int_(neighbors.shape[0] / 2), :, :]
      temporal_feat_t = temporal_feat[np.int_(neighbors.shape[0] / 2):, :, :]

      SIP_encoding_filtered, MIP_encoding_filtered = self.filter(SIP_encoding, MIP_encoding, temporal_feat_s, temporal_feat_t)

      combined_features_SIP = self.mlp_mixer(input_tensor=temporal_feat,
                                             guide_tensor=SIP_encoding_filtered)
      combined_features_MIP = self.mlp_mixer(input_tensor=temporal_feat,
                                             guide_tensor=MIP_encoding_filtered)
      node_embeddings = self.pooler(combined_features_SIP, combined_features_MIP)

      return node_embeddings

def get_embedding_module(module_type, node_features, edge_features, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                        dropout=0.1, n_neighbors=None,n_filters=None
                         ):
  if module_type == "STG":
    return STG(node_features=node_features,
                                    edge_features=edge_features,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    dropout=dropout, n_neighbors=n_neighbors,n_filters=n_filters)

  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))

class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, diset_neighbor_samplermension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)

class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,dropout=dropout)
        self.guide_mixer = nn.Linear(num_channels * 2, num_channels)

    def forward(self, input_tensor: torch.Tensor, guide_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # guide_tensor = torch.repeat_interleave(guide_tensor.unsqueeze(1),input_tensor.size(1),dim=1)
        x = torch.cat((input_tensor, guide_tensor), dim=2)
        input_tensor = self.guide_mixer(x)
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor


class Pooler(nn.Module):
    def __init__(self, embed_dim):
        super(Pooler, self).__init__()
        self.fc1 = nn.Sequential(
          nn.Linear(in_features=embed_dim + embed_dim, out_features=embed_dim),
          nn.ReLU(),
          nn.Linear(in_features=embed_dim, out_features=embed_dim))

    def forward(self, x1, x2):
        output1 = torch.mean(x1, dim=1)
        output2 = torch.mean(x2, dim=1)
        output = self.fc1(torch.cat([output1, output2],dim=-1))
        return output

class FilterLayer(nn.Module):
    def __init__(self, max_input_length: int, hidden_dim: int,filter_num: int, weight):
        super(FilterLayer, self).__init__()
        self.max_input_length = max_input_length
        self.complex_weight = weight
        self.Dropout = nn.Dropout(0.1)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=hidden_dim + hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden_dim, out_features=filter_num),
            nn.Dropout(0.1))

        self.LayerNorm1 = nn.LayerNorm(hidden_dim * 2)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)

    def forward(self, input_tensor1: torch.Tensor,input_tensor2: torch.Tensor, input_tensor3: torch.Tensor, input_tensor4: torch.Tensor):

        batch, seq_len, hidden = input_tensor1.shape

        input_tensor3 = torch.mean(input_tensor3, dim=1)
        input_tensor4 = torch.mean(input_tensor4, dim=1)
        X_input = torch.repeat_interleave(torch.cat([input_tensor3, input_tensor4], dim=-1).unsqueeze(1), 2,
                                              dim=1).view(batch, -1)

        X_input = self.LayerNorm1(X_input)
        filter_weight = self.fc1(X_input).softmax(dim=-1).unsqueeze(1).unsqueeze(1).unsqueeze(4)
        complex_weight_sum = torch.sum(filter_weight * self.complex_weight, dim=-2)
        sigmoid_weight = torch.sigmoid(complex_weight_sum)

        hidden_states_a = input_tensor1
        hidden_states_a_fft = torch.fft.rfft(hidden_states_a, n=self.max_input_length, dim=1, norm='forward')
        weight_a = torch.view_as_complex(sigmoid_weight)
        hidden_states_a_fft = hidden_states_a_fft * weight_a
        sequence_emb_a = torch.fft.irfft(hidden_states_a_fft, n=self.max_input_length, dim=1, norm='forward')
        sequence_emb_a = sequence_emb_a[:, 0:seq_len, :]
        sequence_emb_a = self.Dropout(sequence_emb_a)
        sequence_emb_a = sequence_emb_a + input_tensor1
        sequence_emb_a = self.LayerNorm2(sequence_emb_a)

        hidden_states_b = input_tensor2
        hidden_states_b_fft = torch.fft.rfft(hidden_states_b, n=self.max_input_length, dim=1, norm='forward')
        weight_b = 1 - weight_a
        hidden_states_b_fft = hidden_states_b_fft * weight_b
        sequence_emb_b = torch.fft.irfft(hidden_states_b_fft, n=self.max_input_length, dim=1, norm='forward')
        sequence_emb_b = sequence_emb_b[:, 0:seq_len, :]
        sequence_emb_b = self.Dropout(sequence_emb_b)
        sequence_emb_b = sequence_emb_b + input_tensor2
        sequence_emb_b = self.LayerNorm2(sequence_emb_b)

        return sequence_emb_a, sequence_emb_b


