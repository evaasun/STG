
# Structure-Aware Model for Representation Learning on Temporal Graphs

This repository is the official implementation of [Structure-Aware Model for Representation Learning on Temporal Graphs]. 

## Requirements

Python >= 3.8.0

Pytorch >= 2.2.0

## Dataset

- The dynamic graph datasets come from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg), which can be download [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Please put the datasets into the `DG_data` folder.
- Please run `processed_data/preprocess_data.py` for pre-processing the datasets.


## Running

To train the model STG in the paper with default settings, run this command:

```
python train_link_prediction.py --dataset_name wikipedia --num_neighbors 10 --num_filters 4 --gpu 0
```
To evaluate the model STG in the paper with different negative sampling strategies (random/historical/inductive), run this command:

```
python evaluate_link_prediction.py --dataset_name wikipedia --negative_sample_strategy historical --num_neighbors 10 --num_filters 4 --gpu 0
```

## Acknowledgment
We sincerely appreciate the authors of the open-source codes and datasets. They make this community wonderful.
