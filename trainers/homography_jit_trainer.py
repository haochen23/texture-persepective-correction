import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.tf_logger import TFLogger
from utils.txt_logger import create_logger
from config import Config, homography_config
from data import get_homography_jit_loader
from models.model_homography import HomographyNet
import os
from utils.image_utils import get_train_val_paths
import s3fs

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class HomographyNetTrainer:

    def __init__(self, conf):
        """Constructor
            Inputs:
                conf:   Config object
        """
        self.cuda = conf.cuda
        self.device = conf.device
        self.seed = conf.seed
        self.lr = conf.lr
        self.epochs = conf.epochs
        self.save_epoch = conf.save_epoch
        self.batch_size = conf.batch_size
        self.log_interval = conf.log_interval
        self.data_dir = conf.data_dir
        self.save_path = conf.save_path
        self.out_len = conf.out_len
        self.apply_norm = conf.apply_norm
        self.norm_type = conf.norm_type
        self.apply_dropout = conf.apply_dropout
        self.drop_out = conf.drop_out
        self.s3_bucket = conf.s3_bucket
        self.restore_model = conf.restore_model
        self.restore_at = conf.restore_at
        self.s3 = s3fs.S3FileSystem(anon=False)

        # create loggers
        self.txt_logger = create_logger("HomographyJIT-Train", "logs/")
        self.tf_logger = TFLogger(r'tensorboard_logs/Homography/')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.manual_seed(self.seed)

        train_paths, val_paths = get_train_val_paths(data_dir=self.data_dir,
                                                     split_ratio=homography_config['validation_split_ratio'])

        self.train_loader = get_homography_jit_loader(image_paths=train_paths,
                                                      batch_size=self.batch_size,
                                                      out_t_len=self.out_len)

        self.val_loader = get_homography_jit_loader(image_paths=val_paths,
                                                    batch_size=self.batch_size,
                                                    out_t_len=self.out_len)

        self.model = HomographyNet(apply_norm=self.apply_norm, norm_type=self.norm_type,
                                   apply_dropout=self.apply_dropout, drop_out=self.drop_out,
                                   out_len=self.out_len)
        self.starting_at = 1
        if self.restore_model:
            self.restore_checkpoint(restore_at=self.restore_at)
        self.criterion = nn.MSELoss()
        self.globaliter = 0

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                     base_lr=self.lr/30,
                                                     max_lr=self.lr,
                                                     step_size_up=2000,
                                                     cycle_momentum=False)

    def train_epoch(self, epoch):
        self.txt_logger.info(f"Training epoch {epoch}...")
        self.model.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if self.cuda:
                images = images.cuda()
                targets = targets.cuda()

            images = Variable(images)
            targets = Variable(targets)

            self.optimizer.zero_grad()
            predicted = self.model(images)
            loss = self.criterion(predicted, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if batch_idx % self.log_interval == 0:
                self.txt_logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

            # ============ TensorBoard logging ============#
            self.globaliter += 1
            info = {'train_loss': loss.item()}
            for tag, value in info.items():
                self.tf_logger.scalar_summary(tag, value, step=self.globaliter)

            # saving model at every epoch
            if self.save_epoch:
                if self.cuda:
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.cpu(), self.save_path + f"epoch-{epoch}.pt")
                    else:
                        torch.save(self.model.cpu(), self.save_path + f"epoch-{epoch}.pt")
                    self.model.cuda()
                else:
                    torch.save(self.model.cpu(), self.save_path + f"epoch-{epoch}.pt")

    def test_epoch(self, epoch):
        self.txt_logger.info(f"Validating epoch {epoch}... ")
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if self.cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                images = Variable(images)
                targets = Variable(targets)
                predicted = self.model(images)
                loss = self.criterion(predicted, targets)
                test_loss += loss.item()
            test_loss /= batch_idx
        self.txt_logger.info('\nTest set Epoch: {} Average loss: {:.4f}, \n'.format(epoch, test_loss))
        info = {'val_loss': test_loss}
        for tag, value in info.items():
            self.tf_logger.scalar_summary(tag, value, step=self.globaliter)
        return test_loss

    def train(self):

        self.txt_logger.info("Training model...")
        best_valid_loss = float('inf')

        # train another self.epochs epochs
        for epoch in range(self.starting_at, self.starting_at + self.epochs + 1):
            self.train_epoch(epoch)
            valid_loss = self.test_epoch(epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.txt_logger.info(f"A better model found, Epoch {epoch}. Saving the best model...")
                # save cpu models
                if self.cuda:
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.cpu(), f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")
                    else:
                        torch.save(self.model.cpu(), f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")
                    self.model.cuda()
                else:
                    torch.save(self.model.cpu(), f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")
                # upload model to s3
                self.s3.put( f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt",
                             's3://' + self.s3_bucket + f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")

    def restore_checkpoint(self, restore_at=None):
        """
        restore checkpoint from s3 bucket or local folder if exists
        Args:
            restore_at:  int, epoch or checkpoint number to be restored

        Returns:

        """
        if restore_at is None:
            try:
                restore_at = int(sorted(self.s3.ls('s3://' + self.s3_bucket + f"{self.save_path}"))[-1].split('_')[-2])
            except IndexError as ex:
                pass

        checkpoint = [_ for _ in self.s3.ls('s3://' + self.s3_bucket + f"{self.save_path}") if f"model_at_{restore_at}_loss" in _]
        if len(checkpoint) > 0:
            checkpoint = checkpoint[0]
            self.starting_at = restore_at
            # if local directory contains the file, load from local
            if os.path.isfile(self.save_path + checkpoint):
                self.model = torch.load(self.save_path + checkpoint)
            else:
                # download checkpoint from s3
                self.s3.get('s3://' + self.s3_bucket + f"{self.save_path}" + checkpoint, self.save_path)
                self.model = torch.load(self.save_path + checkpoint)

        else:
            return







if __name__ == '__main__':
    model_config = Config(
        cuda=True if torch.cuda.is_available() else False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=2,
        lr=0.003,
        epochs=80,
        save_epoch=False,
        batch_size=8,
        log_interval=100,
        data_dir='dataset/processed/',
        save_path='homography_v1/',
        out_len=3,
        apply_dropout=False,
        drop_out=0.4,
        apply_norm=False,
        norm_type="BatchNorm",
        s3_bucket="deeppbrmodels/homography_no_norm_no_drop/",
        restore_model=True,
        restore_at=None
    )
