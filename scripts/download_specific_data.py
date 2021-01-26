import s3fs
import numpy as np

np.random.seed(42)

categories = ["wood",'brick', 'tile', 'floor', 'leave', 'rock']
files_list = []
s3 = s3fs.S3FileSystem(anon=False)
for f in s3.ls("s3://biglook/"):
    for cat in categories:
        if cat in f.lower():
            files_list.append(f)

files_list = sorted(list(set(files_list)))
indices = np.random.permutation(len(files_list))[:560]
new_list = [files_list[i] for i in indices]
files = []
for d in new_list:
    for f in s3.ls(d+'/'):
        if "albedo" in f.lower():
            files.append(f)

for f in files:
    print(f"s3://{f}")
    s3.get(f"s3://{f}", f'dataset/{f}', recursive=True)