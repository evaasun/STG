
# Temporal Graph Representation Learning with Neighborhood Reasoning

This repository is the official implementation of [Temporal Graph Representation Learning with Neighborhood Reasoning]. 

## Requirements

To install requirements:

``conda env create -f environment.yaml
``

## Running

To train and evaluate the model TGNR in the paper with default settings, run this command:

``python main.py --dataset_name uci --negative_sample_strategy random --gpu 0
``

## Dataset

- You can download processed datasets from [DGB](https://github.com/fpour/DGB). Please put the data into the processed\_data folder in the following format: /processed\_data/uci/ml\_uci.csv
- Or you can use customized datasets. Please put *Your\_data* into the DG\_data folder in the following format: /DG\_data/*Your\_data*/*Your\_data*.csv. Then run preprocess_data.py in the processed\_data folder.

## Results

Our model achieves the following performance on the UCI dataset with five runs under transductive settings:

| TGNR | AP  | AUC |
|:----:|:----------------:| :----------------: |
|  UCI | 98.82 ± 0.02 %   | 98.70 ± 0.03 %     |

## Acknowledgment
We sincerely appreciate the authors of the open source codes and datasets used in our paper. They make this community wonderful.
