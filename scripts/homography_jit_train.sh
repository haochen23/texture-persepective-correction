#!/usr/bin/env bash
python homography_jit_train.py \
--epochs 30 --lr 0.003 --batch_size 8 \
--dataset_dir ./dataset/biglook/ --seed 42 \
--save_epoch \
--save_path homography_batchnorm_dropout/ \
--log_interval 5 \
--target_len 3 \
--apply_norm \
--norm_type BatchNorm \
--apply_dropout \
--dropout_ratio 0.4 \
--s3_bucket deeppbrmodels/ \
--restore_model True \
--restore_at  None
