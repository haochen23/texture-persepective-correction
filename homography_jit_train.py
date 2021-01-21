from trainers.homography_jit_trainer import HomographyNetTrainer
import torch
from config import Config
import argparse


parser = argparse.ArgumentParser(description='GeoNetM')
parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
parser.add_argument('--log_interval', type=int, default=10, help="Log information interval")
parser.add_argument('--lr', type=float, default=0.003)
# parser.add_argument('--data_num', type=int, default=50000, metavar='N')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument("--dataset_dir", type=str, default='dataset/biglook/')
parser.add_argument("--save_epoch", default=False, action='store_true')
parser.add_argument("--apply_norm", default=False, action='store_ture')
parser.add_argument("--norm_type", type=str, default="BatchNorm", help="Type of Normalization Layer, ['BatchNorm', 'InstanceNorm']")
parser.add_argument("--apply_dropout", default=False, action="store_ture")
parser.add_argument("--dropout_ratio", type=float, default=0.4, help="Drop out layer ratio")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target_len", type=int, default=3, help="target tensor lenght")
parser.add_argument("--save_path", type=str, default='homography_v1/', help='Model save path.')
parser.add_argument("--s3_bucket", type=str, default="deeppbrmodels/homography_no_norm_no_drop",
                    help="s3 bucket to store the saved models")
args = parser.parse_args()


if __name__ == '__main__':
    model_config = Config(
        cuda=True if torch.cuda.is_available() else False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        save_epoch=args.save_epoch,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        data_dir=args.dataset_dir,
        save_path=args.save_path,
        out_len=args.target_len,
        apply_dropout=args.apply_dropout,
        drop_out=args.dropout_ratio,
        apply_norm=args.apply_norm,
        norm_type=args.norm_type,
        s3_bucket=args.s3_bucket
    )

    trainer = HomographyNetTrainer(model_config)
    trainer.train()
