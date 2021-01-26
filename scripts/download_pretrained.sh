cp scripts/download_pretrained_model.py .

python download_pretrained_model.py --bucket \
s3://deeppbrmodels/homography_batchnorm_dropout/ \
--epoch None \
--save_dir pretrained/ \

rm download_pretrained_model.py

