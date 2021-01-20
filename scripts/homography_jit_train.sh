#!/usr/bin/env bash
python flownet_train.py \
--epochs 80 --lr 0.003 --batch_size 8 \
--dataset_dir ./dataset/biglook/ --seed 42 \
--save_epoch \
--save_path homography_v1/ \
--log_interval 5 \
--target_len 3