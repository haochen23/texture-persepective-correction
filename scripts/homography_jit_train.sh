#!/usr/bin/env bash
python homography_jit_train.py \
--epochs 80 --lr 0.003 --batch_size 8 \
--dataset_dir ./dataset/biglook/ --seed 42 \
--save_epoch \
--save_path homography_batchnorm_dropout/ \
--log_interval 5 \
--target_len 3 \
--apply_norm \
--norm_type BatchNorm \
--apply_dropout \
--dropout_ratio 0.4 \
--s3_bucket deeppbrmodels/homography_batchnorm_dropout/