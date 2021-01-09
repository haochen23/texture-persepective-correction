cp scripts/build_homography_dataset.py .

python build_homography_dataset.py \
--input_dir dataset/biglook/ \
--output_dir dataset/homography/ \
--split_ratio 0.2

rm build_homography_dataset.py