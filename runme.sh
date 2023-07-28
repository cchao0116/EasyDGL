#!/usr/bin/env bash

source activate py37_th

python src/demo_recsys.py

python src/demo_traffic.py

conda deactivate