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

[2023.06.30] We will release the Pytorch-DGL version with ten more dynamic graph models.

[2023.03.16] We release the TensorFlow version of our codes for link prediction.


## Results for Link Prediction
### Dataset
We use the Netflix benchmark to evaluate model performance, where the Tensorflow Record scheme is as follows:

| Feature Name | Feature Type             | Content                           |
|:------------:|:-------------------------|:----------------------------------|
|    seqs_i    | FixedLenFeature(int64)   | sequence of a user's rated items  |
|    seqs_t    | FixedLenFeature(float32) | sequence of timestamps in sec     |
|  seqs_hour   | FixedLenFeature(int64)   | sequence of timestamps in hour    |
|   seqs_day   | FixedLenFeature(int64)   | sequence of timestamps in day     |
| seqs_weekday | FixedLenFeature(int64)   | sequence of timestamps in weekday |
|  seqs_month  | FixedLenFeature(int64)   | sequence of timestamps in month   |

TFRECORD Download:
[Google](https://drive.google.com/file/d/145lWyMn0mFdXwUOOpIdxGo7FoDXfL-dL/view?usp=share_link),
[夸克](https://pan.quark.cn/s/f290b4ff57c4)

### Results

Below we report the HR@50, NDCG@50 and NDCG@100 results *on the above provided dataset*.

| Model           |    HR@50    |   NDCG@50   |  NDCG@100   |
|:----------------|:-----------:|:-----------:|:-----------:|
| GRU4REC         |   0.40903   |   0.18904   |   0.20321   | 
| SASREC          |   0.41802   |   0.19614   |   0.21075   | 
| S2PNM           |   0.41960   |   0.19536   |   0.20991   | 
| BERT4REC        |   0.42487   |   0.19782   |   0.21257   | 
| GREC            |   0.41915   |   0.19573   |   0.20974   |
| TGAT            |   0.41633   |   0.19205   |   0.20679   | 
| TiSASREC        |   0.44583   |   0.20879   |   0.22334   | 
| TimelyREC       |   0.42202   |   0.19897   |   0.21315   | 
| CTSMA           |   0.45240   |   0.21141   |   0.22589   |
| EasyDGL (ours.) | **0.48320** | **0.23104** | **0.24476** |



### Folder Specification

- ```conf/```: configurations for logging
- ```data/```: preprocessing scripts for data filter and split
- ```runme.sh```: train or evaluate EasyDGL and baseline models
- ```src/```: codes for model definition


<details onclose="True">
<summary><b>Supported algorithms:</b></summary>

- [x] [GRU4REC](src/model/GRU4REC.py) (ICLR'2016)
- [x] [SASREC](src/model/SASREC.py) (ICDM'2018)
- [x] [BERT4REC](src/model/BERT4REC.py) (CIKM'2019)
- [x] [GREC](src/model/GREC.py) (WWW'2020)
- [x] [TGAT](src/model/TGAT.py) (ICLR'2020)
- [x] [TiSASREC](src/model/TiSASREC.py) (WSDM'2020)
- [x] [TimelyREC](src/model/TimelyREC.py) (WWW'2021)
- [x] [CTSMA](src/model/CTSMA.py) (ICML'2021)
- [x] [S2PNM](src/model/S2PNM.py) (TKDE'2022)
- [X] [EasyDGL](src/model/EasyDGL.py) (ours.)

</details>

### Run the Code

Download our data to $DATA_HOME directory, 
then Reproduce above results on Netflix benchmark:

``` 
bash runme.sh ${$DATA_HOME}
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
