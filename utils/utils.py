import random
import torch
import torch.nn as nn
import numpy as np

class NegativeEdgeSampler(object):

  def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None,
               last_observed_time: float = None,
               negative_sample_strategy: str = 'random', seed: int = None):
    """
    Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
    :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
    :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
    :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
    :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
    :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
    :param seed: int, random seed
    """
    self.seed = seed
    self.negative_sample_strategy = negative_sample_strategy
    self.src_node_ids = src_node_ids
    self.dst_node_ids = dst_node_ids
    self.interact_times = interact_times
    self.unique_src_node_ids = np.unique(src_node_ids)
    self.unique_dst_node_ids = np.unique(dst_node_ids)
    self.unique_interact_times = np.unique(interact_times)
    self.earliest_time = min(self.unique_interact_times)
    self.last_observed_time = last_observed_time

    if self.negative_sample_strategy != 'random':
      # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
      self.possible_edges = set(
        (src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in
        self.unique_dst_node_ids)

    if self.negative_sample_strategy == 'inductive':
      # set of observed edges
      self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

    if self.seed is not None:
      self.random_state = np.random.RandomState(self.seed)

  def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
    """
    get unique edges happened between start and end time
    :param start_time: float, start timestamp
    :param end_time: float, end timestamp
    :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
    """
    selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
    # return the unique select source and destination nodes in the selected time interval
    return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in
               zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

  def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
             current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
    """
    sample negative edges, support random, historical and inductive sampling strategy
    :param size: int, number of sampled negative edges
    :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
    :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
    :param current_batch_start_time: float, start time in the current batch
    :param current_batch_end_time: float, end time in the current batch
    :return:
    """
    if self.negative_sample_strategy == 'random':
      negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
    elif self.negative_sample_strategy == 'historical':
      negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size,
                                                                            batch_src_node_ids=batch_src_node_ids,
                                                                            batch_dst_node_ids=batch_dst_node_ids,
                                                                            current_batch_start_time=current_batch_start_time,
                                                                            current_batch_end_time=current_batch_end_time)
    elif self.negative_sample_strategy == 'inductive':
      negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size,
                                                                           batch_src_node_ids=batch_src_node_ids,
                                                                           batch_dst_node_ids=batch_dst_node_ids,
                                                                           current_batch_start_time=current_batch_start_time,
                                                                           current_batch_end_time=current_batch_end_time)
    else:
      raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
    return negative_src_node_ids, negative_dst_node_ids

  def random_sample(self, size: int):
    """
    random sampling strategy, which is used by previous works
    :param size: int, number of sampled negative edges
    :return:
    """
    if self.seed is None:
      random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
      random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
    else:
      random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
      random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
    return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[
      random_sample_edge_dst_node_indices]

  def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray,
                                         batch_dst_node_ids: np.ndarray):
    """
    random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
    used for historical and inductive sampling strategy
    :param size: int, number of sampled negative edges
    :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
    :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
    :return:
    """
    assert batch_src_node_ids is not None and batch_dst_node_ids is not None
    batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in
                      zip(batch_src_node_ids, batch_dst_node_ids))
    possible_random_edges = list(self.possible_edges - batch_edges)
    assert len(possible_random_edges) > 0
    # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
    random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size,
                                                   replace=len(possible_random_edges) < size)
    return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
           np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

  def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                        current_batch_start_time: float, current_batch_end_time: float):
    """
    historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
    if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
    :param size: int, number of sampled negative edges
    :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
    :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
    :param current_batch_start_time: float, start time in the current batch
    :param current_batch_end_time: float, end time in the current batch
    :return:
    """
    assert self.seed is not None
    # get historical edges up to current_batch_start_time
    historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                    end_time=current_batch_start_time)
    # get edges in the current batch
    current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                       end_time=current_batch_end_time)
    # get source and destination node ids of unique historical edges
    unique_historical_edges = historical_edges - current_batch_edges
    unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
    unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

    # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
    if size > len(unique_historical_edges):
      num_random_sample_edges = size - len(unique_historical_edges)
      random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
        size=num_random_sample_edges,
        batch_src_node_ids=batch_src_node_ids,
        batch_dst_node_ids=batch_dst_node_ids)

      negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
      negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
    else:
      historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size,
                                                                     replace=False)
      negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
      negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

    # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
    # Hence, convert the type to long to guarantee valid index
    return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

  def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                       current_batch_start_time: float, current_batch_end_time: float):
    """
    inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
    if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
    :param size: int, number of sampled negative edges
    :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
    :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
    :param current_batch_start_time: float, start time in the current batch
    :param current_batch_end_time: float, end time in the current batch
    :return:
    """
    assert self.seed is not None
    # get historical edges up to current_batch_start_time
    historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                    end_time=current_batch_start_time)
    # get edges in the current batch
    current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                       end_time=current_batch_end_time)
    # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
    unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
    unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
    unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

    # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
    if size > len(unique_inductive_edges):
      num_random_sample_edges = size - len(unique_inductive_edges)
      random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
        size=num_random_sample_edges,
        batch_src_node_ids=batch_src_node_ids,
        batch_dst_node_ids=batch_dst_node_ids)

      negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
      negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
    else:
      inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size,
                                                                    replace=False)
      negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
      negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

    # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
    # Hence, convert the type to long to guarantee valid index
    return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

  def reset_random_state(self):
    """
    reset the random state by self.seed
    :return:
    """
    self.random_state = np.random.RandomState(self.seed)

class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)

def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)

class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]


    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:

        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]
          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

