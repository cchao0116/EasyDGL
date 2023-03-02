## EasyDGL: an Encode, Train and Interpret Pipeline for Continuous-time Graph Learning

The official implementation for "EasyDGL: an Encode, Train and Interpret Pipeline for Continuous-time Graph Learning".

## What's news

[2023.03.02] We release the early version of our codes for link prediction.

### Datasets

We use the Netflix benchmark for link prediction:

| Feature Name | Feature Type             | Content                             |
|:------------:|:-------------------------|:------------------------------------|
|    seqs_i    | FixedLenFeature(int64)   | sequence of a user's rated items    |
|    seqs_t    | FixedLenFeature(float32) | sequence of timestamps in sec       |
|  seqs_hour   | FixedLenFeature(int64)   | sequence of timestamps in hour      |
|   seqs_day   | FixedLenFeature(int64)   | sequence of timestamps in day       |
| seqs_weekday | FixedLenFeature(int64)   | sequence of timestamps in week      |
|  seqs_month  | FixedLenFeature(int64)   | sequence of timestamps in month     |

**TFRECORD Download**:
[Google](https://ogb.stanford.edu/),
[Baidu](https://ogb.stanford.edu/)

### Results

## How to run our codes?

### Folder Specification

- ```conf/```: configurations for backbone (GCN, GIN, GraphSAGE)
- ```data/```: three trained model on OGB-BACE datasets
- ```src/```: preprocessing scripts for data and model definition
- ```runme.sh```: train or evaluate our model on OGB benchmark

### Run the Code

Train the baselines on OGB benchmark:

``` 
python baseline_ogb.py --dataset ogbg-molbace --gnn gcn --device ${device} --seed ${seed}
```

### Citation

If you find our codes useful, please consider citing our work

```bibtex
@inproceedings{chen2021learning,
  title={Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation},
  author={Chen, Chao and Geng, Haoyu and Yang, Nianzu and Yan, Junchi and Xue, Daiyue and Yu, Jianping and Yang, Xiaokang},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML '21)},
  pages={1606--1616},
  year={2021},
  organization={PMLR}
}
```
