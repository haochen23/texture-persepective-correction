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
from utils.image_utils import (get_train_val_paths,
                               plot_grad_flow,
                               tensor2im,
                               plot_images2fig,
                               figure2image)
from utils.homography_utils import pad_and_crop_to_size, decode_output
import s3fs
import random
import cv2
import numpy as np

from PIL import ImageFile, Image

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
        self.txt_logger_file = conf.txt_logger

        # create loggers
        self.txt_logger = create_logger(self.txt_logger_file, "logs/")
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                     base_lr=self.lr / 50,
                                                     max_lr=self.lr,
                                                     step_size_up=2000,
                                                     cycle_momentum=False)
        self.scheduler = self.scheduler
        self.starting_at = 1
        if self.restore_model:
            self.restore_checkpoint(restore_at=self.restore_at)

        # self.criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.L1Loss(reduction='sum')
        self.globaliter = 0

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

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
            self.txt_logger.info(f" True vs Predicted value for batch {batch_idx}")
            self.txt_logger.info(targets)
            self.txt_logger.info(predicted)
            loss0 = self.criterion(predicted[:, 0], targets[:, 0])
            loss1 = self.criterion(predicted[:, 1], targets[:, 1])
            loss2 = self.criterion(predicted[:, 2], targets[:, 2])
            loss = loss0 + loss1 * 10000 + loss2 * 10000
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if batch_idx % self.log_interval == 0:
                self.txt_logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
                # plot_grad_flow(self.model.named_parameters())
                if len(targets) > 1:
                    indices = random.choices(range(len(targets)), k=2)
                else:
                    indices = [0]

                self.save_output(im_tensors=images[indices],
                                 predictions=predicted[indices],
                                 targets=targets[indices])


            # ============ TensorBoard logging ============#
            self.globaliter += 1
            info = {'train_loss': loss.item()}
            for tag, value in info.items():
                self.tf_logger.scalar_summary(tag, value, step=self.globaliter)

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
                # loss = self.criterion(predicted, targets)
                loss0 = self.criterion(predicted[:, 0], targets[:, 0])
                loss1 = self.criterion(predicted[:, 1], targets[:, 1])
                loss2 = self.criterion(predicted[:, 2], targets[:, 2])
                loss = loss0 + loss1 * 10000 + loss2 * 10000
                test_loss += loss.item()
            test_loss /= batch_idx
        self.txt_logger.info('\nTest set Epoch: {} Average loss: {:.4f}, \n'.format(epoch, test_loss))
        info = {'val_loss': test_loss}
        for tag, value in info.items():
            self.tf_logger.scalar_summary(tag, value, step=self.globaliter)

        # saving model at every epoch
        if self.save_epoch:
            self.save_checkpoint(on_cuda=self.cuda,
                                 model_path=self.save_path,
                                 epoch_num=epoch,
                                 loss_value=test_loss)

        return test_loss

    def save_checkpoint(self, on_cuda, model_path, epoch_num, loss_value):
        if on_cuda:
            if isinstance(self.model, nn.DataParallel):
                torch.save({'model_state_dict': self.model.module.cpu().state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'lr_scheduler_state_dict': self.scheduler.state_dict(),
                            'epoch': self.starting_at,
                            'loss': loss_value,
                            'globaliter': self.globaliter},
                           f"{model_path}model_at_{epoch_num}_loss({loss_value}).pt")
            else:
                torch.save({'model_state_dict': self.model.cpu().state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'lr_scheduler_state_dict': self.scheduler.state_dict(),
                            'epoch': self.starting_at,
                            'loss': loss_value,
                            'globaliter': self.globaliter},
                           f"{model_path}model_at_{epoch_num}_loss({loss_value}).pt")
            self.model.cuda()
        # Upload to s3
        self.s3.put(f"{model_path}model_at_{epoch_num}_loss({loss_value}).pt",
                    's3://' + self.s3_bucket + f"{model_path}model_at_{epoch_num}_loss({loss_value}).pt")
        os.remove(f"{model_path}model_at_{epoch_num}_loss({loss_value}).pt")

    def train(self):

        self.txt_logger.info("Training model...")
        best_valid_loss = float('inf')

        # train another self.epochs epochs
        for epoch in range(self.starting_at, self.starting_at + self.epochs + 1):
            self.train_epoch(epoch)
            if epoch == self.starting_at:
                self.tf_logger.writer.add_graph(self.model, input_to_model=torch.rand([8, 3, 512, 512], device="cuda"))
            valid_loss = self.test_epoch(epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.txt_logger.info(f"A better model found, Epoch {epoch}. Saving the best model...")
                # save cpu models
                # if self.cuda:
                #     if isinstance(self.model, nn.DataParallel):
                #         torch.save(self.model.module.cpu(), f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")
                #     else:
                #         torch.save(self.model.cpu(), f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")
                #     self.model.cuda()
                # else:
                #     torch.save(self.model.cpu(), f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")
                # upload model to s3
                # self.s3.put( f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt",
                #              's3://' + self.s3_bucket + f"{self.save_path}model_at_{epoch}_loss({best_valid_loss}).pt")

    def restore_checkpoint(self, restore_at=None):
        """
        restore checkpoint from s3 bucket or local folder if exists
        Args:
            restore_at:  int, epoch or checkpoint number to be restored

        Returns:

        """
        if restore_at is None:
            try:
                restore_at = int(sorted(self.s3.ls('s3://' + self.s3_bucket + f"{self.save_path}"),
                                        key=lambda x: int(x.split('_')[-2]))[-1].split('_')[-2])
            except Exception as ex:
                self.txt_logger.error("Checkpoints not found in S3.")
                return

        checkpoint_file = [_ for _ in self.s3.ls('s3://' + self.s3_bucket + f"{self.save_path}") if f"model_at_{restore_at}_loss" in _]
        if len(checkpoint_file) > 0:
            checkpoint_file = checkpoint_file[0]
            self.txt_logger.info(f"Restoring {checkpoint_file}")
            self.starting_at = restore_at + 1
            # if local directory contains the file, load from local
            if os.path.isfile(self.save_path + checkpoint_file):
                checkpoint = torch.load(self.save_path + checkpoint_file, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                self.globaliter = checkpoint['globaliter']

            else:
                # download checkpoint from s3
                self.s3.get(checkpoint_file, self.save_path + checkpoint_file)
                checkpoint = torch.load(self.save_path + checkpoint_file, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                self.globaliter = checkpoint['globaliter']

        else:
            return

    def save_output(self, im_tensors, predictions, targets):
        images = tensor2im(im_tensors)
        for i in range(len(images)):
            padded_image = np.array(pad_and_crop_to_size(Image.fromarray(images[i].squeeze()), to_size=homography_config['load_width']))
            true_H = decode_output(targets[i].squeeze(),
                                   width=homography_config['load_width'],
                                   height=homography_config['load_height'],
                                   scale=1.0)
            predicted_H = decode_output(predictions[i].squeeze(),
                                        width=homography_config['load_width'],
                                        height=homography_config['load_height'],
                                        scale=1.0)

            true_image = cv2.warpPerspective(padded_image, true_H,
                                             (homography_config['load_width'], homography_config['load_height']))
            predicted_image = cv2.warpPerspective(padded_image, predicted_H,
                                                  (homography_config['load_width'], homography_config['load_height']))

            fig = plot_images2fig(orig_image=padded_image,
                                  true_image=true_image,
                                  predicted_image=predicted_image)
            self.tf_logger.writer.add_figure(f"Global Step {self.globaliter} - Exxample {i}",
                                             figure=fig,
                                             global_step=self.globaliter)


if __name__ == '__main__':
    model_config = Config(
        cuda=True if torch.cuda.is_available() else False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        lr=0.03,
        epochs=500,
        save_epoch=True,
        batch_size=16,
        log_interval=5,
        data_dir='dataset/biglook/',
        save_path='homography_mutlihead_nosigmoid_bs16/',
        out_len=3,
        apply_dropout=False,
        drop_out=0.4,
        apply_norm=True,
        norm_type="BatchNorm",
        s3_bucket="deeppbrmodels/",
        restore_model=True,
        restore_at=None,
        txt_logger='homography_multihead_nosigmoid_bs16'
    )
    self = HomographyNetTrainer(model_config)
    self.train()
