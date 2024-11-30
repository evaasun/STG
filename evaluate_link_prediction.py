import logging
import time
import math
import shutil
import json
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse

from model.STG import STG
from utils.dataloader import get_data
from utils.utils import get_neighbor_finder, NegativeEdgeSampler
from utils.EarlyStopping import EarlyStopping
from evaluate_models import eval_edge_prediction
from model.MergeLayer import MergeLayer


torch.set_printoptions(profile="full")


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



### Argument and global variables
parser = argparse.ArgumentParser('Interface for the link prediction task')
parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='uci',
                    choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'uci'])
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--model_name', type=str, default='STG')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'],
                    help='name of optimizer')
parser.add_argument('--negative_sample_strategy', type=str, default='inductive',
                    choices=['random', 'historical', 'inductive'], help='strategy for the negative edge sampling')
parser.add_argument('--num_neighbors', type=int, default=10, help='number of neighbors to sample for each node')
parser.add_argument('--num_filters', type=int, default=4, help='number of neighbors to sample for each node')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--uniform_sample_neighbor_strategy', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')

try:
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
except:
    parser.print_help()
    sys.exit()

# get data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    get_data(args.dataset_name,
             different_new_nodes_between_val_and_test=args.different_new_nodes,
             randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform_sample_neighbor_strategy)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform_sample_neighbor_strategy)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = NegativeEdgeSampler(train_data.sources, train_data.destinations)
#
val_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps,
                                       last_observed_time=train_data.timestamps[-1],
                                       negative_sample_strategy=args.negative_sample_strategy, seed=0)
nn_val_rand_sampler = NegativeEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                          new_node_val_data.timestamps,
                                          last_observed_time=train_data.timestamps[-1],
                                          negative_sample_strategy=args.negative_sample_strategy, seed=1)
test_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps,
                                        last_observed_time=val_data.timestamps[-1],
                                        negative_sample_strategy=args.negative_sample_strategy, seed=2)
nn_test_rand_sampler = NegativeEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations,
                                           new_node_test_data.timestamps,
                                           last_observed_time=val_data.timestamps[-1],
                                           negative_sample_strategy=args.negative_sample_strategy, seed=3)

val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []


for i in range(args.num_runs):
    set_random_seed(seed=i)

    args.seed = i
    args.save_model_name = f'{args.model_name}_seed{args.seed}'
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(
        f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run {i + 1} starts. **********")

    logger.info(f'configuration is {args}')

    # Initialize Model
    model = STG(neighbor_finder=train_ngh_finder, node_features=node_features,
                edge_features=edge_features, device=args.device,
                n_layers=args.n_layer, dropout=args.dropout,
                time_dimension=args.time_dim,
                n_neighbors=args.num_neighbors,
                n_filters=args.num_filters)

    link_predictor = MergeLayer(dim1=node_features.shape[1], dim2=node_features.shape[1],
                                dim3=node_features.shape[1], dim4=1)
    model_all = nn.Sequential(model, link_predictor)

    logger.info(f'model -> {model_all}')
    logger.info(
        f'model name: {args.model_name}, #parameters: {sum([p.numel() for p in model_all.parameters() if p.requires_grad]) * 4} B, '
        f'{sum([p.numel() for p in model_all.parameters() if p.requires_grad]) * 4 / 1024} KB, {sum([p.numel() for p in model_all.parameters() if p.requires_grad]) * 4 / 1024 / 1024} MB.')
    optimizer = torch.optim.Adam(model_all.parameters(), lr=args.learning_rate)
    model_all = model_all.to(args.device)
    args.negative_sample_strategy = 'random'
    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{args.negative_sample_strategy}/{args.num_neighbors}"
    # shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                   save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

    criterion = nn.BCELoss()
    # load the best model
    early_stopping.load_checkpoint(model_all)

    # evaluate the best model
    logger.info(f'get final performance on dataset {args.dataset_name}...')

    ### Test
    model_all[0].embedding_module.neighbor_finder = full_ngh_finder

    test_losses, test_metrics = eval_edge_prediction(model=model_all,
                                                     negative_edge_sampler=test_rand_sampler,
                                                     data=test_data,
                                                     n_neighbors=args.num_neighbors,n_filters = args.num_filters, loss_func=criterion)


    # Test on unseen nodes
    new_node_test_losses, new_node_test_metrics = eval_edge_prediction(model=model_all,
                                                                       negative_edge_sampler=nn_test_rand_sampler,
                                                                       data=new_node_test_data,
                                                                       n_neighbors=args.num_neighbors,n_filters = args.num_filters,
                                                                       loss_func=criterion)

    # store the evaluation metrics at the current run
    val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

    logger.info(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
        logger.info(f'test {metric_name}, {average_test_metric:.4f}')
        test_metric_dict[metric_name] = average_test_metric

    logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
    for metric_name in new_node_test_metrics[0].keys():
        average_new_node_test_metric = np.mean(
            [new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
        logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        new_node_test_metric_dict[metric_name] = average_new_node_test_metric

    single_run_time = time.time() - run_start_time
    logger.info(f'Run {i + 1} cost {single_run_time:.2f} seconds.')

    test_metric_all_runs.append(test_metric_dict)
    new_node_test_metric_all_runs.append(new_node_test_metric_dict)

    # avoid the overlap of logs
    if i < args.num_runs - 1:
        logger.removeHandler(fh)
        logger.removeHandler(ch)

    result_json = {
        "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in
                                  new_node_test_metric_dict}
    }
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/{args.negative_sample_strategy}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)
# store the average metrics at the log of the last run
logger.info(f'metrics over {args.num_runs} runs:')
for metric_name in test_metric_all_runs[0].keys():
    logger.info(
        f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
    logger.info(
        f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

for metric_name in new_node_test_metric_all_runs[0].keys():
    logger.info(
        f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
    logger.info(
        f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
        f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')



