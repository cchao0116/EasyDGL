<h1 align="center"><b>EasyDGL</b></h1>
<p align="center">
    <a href="https://arxiv.org/abs/2303.12341" target="_blank"><img src="http://img.shields.io/badge/cs.LG-arXiv%3A2303.12341-B31B1B.svg" /></a>
    <a href="https://proceedings.mlr.press/v139/chen21h.html"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2721&color=blue"></a>
    <a href="https://github.com/cchao0116/EasyDGL/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/cchao0116/EasyDGL?color=green"></a>
    <a href="https://github.com/cchao0116/EasyDGL/stargazers"><img src="https://img.shields.io/github/stars/cchao0116/EasyDGL?color=yellow&label=Star" alt="Stars"></a>
</p>

The official implementation for
["EasyDGL: Encode, Train and Interpret for Continuous-time Dynamic Graph Learning"](https://arxiv.org/abs/2303.12341).

<div align=center>
    <img src="docs/overview.png"/>
</div>

## What's news

[2023.07.10] We release the pre-version of Pytorch-DGL.


## Results for Link Prediction
### Results

Below we report the HR@50, NDCG@50 and NDCG@100 results *on the above provided dataset*.

| Model                |  HR@50  | NDCG@50 | NDCG@100 |
|:---------------------|:-------:|:-------:|:--------:|
| EasyDGL (Tensorflow) | 0.48320 | 0.23104 | 0.24476  |
| EasyDGL (Torch)      | 0.48252 | 0.23055 | 0.24378  |


### Folder Specification

- ```conf/```: configurations for logging
- ```data/```: dataset for model training and evaluation
- ```runme.sh```: train or evaluate EasyDGL and baseline models
- ```src/```: codes for model definition


### Run the Code

Download our data to $DATA_HOME directory, 
then Reproduce above results on Netflix benchmark:

``` 
bash runme.sh
```

## Citation

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

@article{chen2023easydgl,
  title={EasyDGL: Encode, Train and Interpret for Continuous-time Dynamic Graph Learning},
  author={Chen, Chao and Geng, Haoyu and Yang, Nianzu and Yang, Xiaokang and Yan, Junchi},
  journal={arXiv preprint arXiv:2303.12341},
  year={2023}
}
```
