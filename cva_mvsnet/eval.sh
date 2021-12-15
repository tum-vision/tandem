#!/bin/bash

python eval.py --batch_size 1 --data_dir data/replica --pose_ext gt \
  --tuples_ext dso_optimization_windows_last3 pretrained/ablation/abl01_baseline.ckpt

python eval.py --batch_size 1 --data_dir data/replica --pose_ext gt \
  --tuples_ext dso_optimization_windows pretrained/ablation/abl02_vo_window.ckpt

python eval.py --batch_size 1 --data_dir data/replica --pose_ext gt \
  --tuples_ext dso_optimization_windows pretrained/ablation/abl03_view_aggregation.ckpt

python eval.py --batch_size 1 --data_dir data/replica --pose_ext gt \
  --tuples_ext dso_optimization_windows pretrained/ablation/abl04_fewer_depth_planes.ckpt
