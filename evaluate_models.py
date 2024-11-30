
from utils.metrics import get_link_prediction_metrics
import math
import torch
import numpy as np
import os
def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, n_filters,loss_func, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  evaluate_losses, evaluate_metrics = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      if negative_edge_sampler.negative_sample_strategy != 'random':
          batch_neg_src_node_ids, dst_negative_samples = negative_edge_sampler.sample(size=size,batch_src_node_ids=sources_batch,
                                                                                            batch_dst_node_ids=destinations_batch,
                                                                                            current_batch_start_time=timestamps_batch[0],
                                                                                            current_batch_end_time=timestamps_batch[-1])
      else:
          _, dst_negative_samples = negative_edge_sampler.sample(size)
          batch_neg_src_node_ids = sources_batch

      batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings  = \
          model[0].compute_edge_probabilities(source_nodes=batch_neg_src_node_ids,
                                                            destination_nodes=dst_negative_samples,
                                                            edge_times=timestamps_batch,
                                                            edge_idxs=None,
                                                            n_neighbors=n_neighbors,n_filters = n_filters)

      # get temporal embedding of source and destination nodes
      # two Tensors, with shape (batch_size, node_feat_dim)
      batch_src_node_embeddings, batch_dst_node_embeddings = \
          model[0].compute_edge_probabilities(source_nodes=sources_batch,
                                                            destination_nodes=destinations_batch,
                                                            edge_times=timestamps_batch,
                                                            edge_idxs=edge_idxs_batch,
                                                            n_neighbors=n_neighbors,n_filters = n_filters)
      positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
      negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

      predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
      labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

      loss = loss_func(input=predicts, target=labels)
      evaluate_losses.append(loss.item())
      evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

  return evaluate_losses, evaluate_metrics



