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

        # create loggers
        self.txt_logger = create_logger("HomographyJIT-Train", "logs/")
        self.tf_logger = TFLogger(r'tensorboard_logs/Homography/')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.manual_seed(self.seed)

        train_paths, val_paths = get_train_val_paths(data_dir=self.data_dir,
                                                     split_ratio=homography_config['validation_split_ratio'])

        self.train_loader = get_homography_jit_loader(image_paths=train_paths,
                                                      batch_size=5,
                                                      out_t_len=3)

        self.val_loader = get_homography_jit_loader(image_paths=val_paths,
                                                    batch_size=5,
                                                    out_t_len=3)

        self.model = HomographyNet(out_len=self.out_len)
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
                                                     step_size_up=2000)

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

        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            valid_loss = self.test_epoch(epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.txt_logger.info(f"A better model found, Epoch {epoch}. Saving the best model...")
                # save cpu models
                if self.cuda:
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.cpu(), f"{self.save_path}model_at_{epoch}.pt")
                    else:
                        torch.save(self.model.cpu(), f"{self.save_path}model_at_{epoch}.pt")
                    self.model.cuda()
                else:
                    torch.save(self.model.cpu(), f"{self.save_path}model_at_{epoch}.pt")



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
        out_len=3
    )
