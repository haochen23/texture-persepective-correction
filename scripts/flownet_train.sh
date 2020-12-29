#!/usr/bin/env bash
python flownet_train.py \
--epochs 50 --lr 0.0003 --batch_size 16 \
--dataset_dir ./dataset/processed/ --seed 42 \
--save_epoch False --save_path output \
--log_interval 5