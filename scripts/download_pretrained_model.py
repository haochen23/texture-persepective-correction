import s3fs
import argparse
import os
from utils import none_or_int

s3 = s3fs.S3FileSystem(anon=False)


parser = argparse.ArgumentParser(description='Download Pretrained Model')
parser.add_argument('--epoch', type=none_or_int, default=None,
                    help="the epoch number that the desired model were saved at")
parser.add_argument('--bucket', type=str, required=True,
                    help="The s3 bucket that stores all the models")
parser.add_argument("--download_all", default=False, action='store_true')
parser.add_argument("--save_dir", type=str, default="pretrained/",
                    help="Directory to save the downloaded model(s)")

args = parser.parse_args()


def download_best_model(bucket, save_dir):
    model_files = [_ for _ in s3.ls(bucket) if _.endswith('.pt')]
    if len(model_files) > 0:
        model_files = sorted(model_files, key=lambda x: int(x.split('_')[-2]))
        model_file = model_files[-1]
        print(f"Downloading {model_file} from {bucket}")
        s3.get(f"s3://{model_file}", f'{save_dir}{model_file}', recursive=True)
    else:
        print(f"No models available in the provided s3 bucket: {bucket}")


def download_model(bucket, download_all=False, epoch=None, save_dir='pretrained/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if download_all:
        model_files = [_ for _ in s3.ls(bucket) if _.endswith('.pt')]
        if len(model_files) > 0:
            for f in model_files:
                print(f"Downloading {f} from {bucket}")
                s3.get(f"s3://{f}", f'{save_dir}{f}', recursive=True)
        else:
            print(f"No models available in the provided s3 bucket: {bucket}")
    else:
        if epoch is not None:
            model_files = [_ for _ in s3.ls(bucket) if int(_.split('_')[-2]) == int(epoch)]
            if len(model_files) > 0:
                model_file = model_files[0]
                print(f"Downloading {model_file} from {bucket}")
                s3.get(f"s3://{model_file}", f'{save_dir}{model_file}', recursive=True)
            else:
                print(f"Specified epoch {epoch} model does not exist. Trying to download the best one from {bucket}")
                download_best_model(bucket=bucket, save_dir=save_dir)
        else:
            print(f"Trying to download the best one from {bucket}")
            download_best_model(bucket=bucket, save_dir=save_dir)


if __name__ == '__main__':
    bucket = args.bucket
    save_dir = args.save_dir
    epoch = args.epoch
    download_all = args.download_all

    download_model(bucket=bucket,
                   download_all=download_all,
                   epoch=epoch,
                   save_dir=save_dir)