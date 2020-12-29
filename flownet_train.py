from trainers.flownet_trainer import FlowNetTrainer
from config import Config
import torch
import argparse

parser = argparse.ArgumentParser(description='GeoNetM')
parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
parser.add_argument('--log_interval', type=int, default=10, help="Log information interval")
parser.add_argument('--lr', type=float, default=0.0003)
# parser.add_argument('--data_num', type=int, default=50000, metavar='N')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument("--dataset_dir", type=str, default='dataset/processed/')
parser.add_argument("--save_epoch", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--layers", type=list, default=[1,1,1,1,2])
parser.add_argument("--save_path", type=str, default='output/', help='Model save path.')
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
        layers=args.layers,
        data_dir=args.dataset_dir,
        save_path=args.save_path
    )

    trainer = FlowNetTrainer(model_config)
    trainer.train()