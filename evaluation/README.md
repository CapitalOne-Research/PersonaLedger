# Evaluation and Benchmarking
We'd like to express gratitude to the authors of this library [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Our evaluation directly adopts their implementation, with our own data loaders implementation.

 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. Download the data from [this link](https://huggingface.co/datasets/capitalone/PersonaLedger), or use the curated benchmarking suite in this repository [`../generation/generated_transactions/raw_generation`](../generation/generated_transactions/raw_generation). Put the data under `dataset` so that the file structure looks like
```
| - evaluation
  | - data_provider
  | - exp
  | - layers
  | - ...
  | - dataset
    | - identity_theft_1months
      | - train.parquet
      | - test.parquet
    | - illiquidity_prediction_1months
      | - train.parquet
      | - test.parquet
      | - labels.json
```

3. Run evaluation. For example,
```
bash ./scripts/identity_1month/Autoformer.sh
```

## Citation of Time-Series-Library

```
@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```
