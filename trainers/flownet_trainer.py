import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.tf_logger import TFLogger
from utils.txt_logger import create_logger
from config import Config
from data.dataset_flownet import get_loader
from models.model_flownet import FlowNet
from models.modules.loss import EPELoss
import os


class FlowNetTrainer:
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
        self.layers = conf.layers
        self.txt_logger = create_logger("FlowNet-Train", "logs/")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.manual_seed(self.seed)

        # kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
        self.train_loader = get_loader(imageDir=self.data_dir + 'train/distorted',
                                       flowDir=self.data_dir + 'train/flow/',
                                       batch_size=self.batch_size)
        self.val_loader = get_loader(imageDir=self.data_dir + 'val/distorted/',
                                     flowDir=self.data_dir + 'val/flow/',
                                     batch_size=self.batch_size)
        self.model = FlowNet(layers=self.layers)
        self.criterion = EPELoss()
        self.globaliter = 0

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.tf_logger = TFLogger(r'tensorboard_logs/')

    def train_epoch(self, epoch):
        self.txt_logger.info(f"Training epoch {epoch}...")
        self.model.train()

        for batch_idx, (images, flow_xs, flow_ys) in enumerate(self.train_loader):
            if self.cuda:
                images = images.cuda()
                flow_xs = flow_xs.cuda()
                flow_ys = flow_ys.cuda()


            images = Variable(images)
            labels_x = Variable(flow_xs)
            labels_y = Variable(flow_ys)
            flow_truth = torch.cat([labels_x, labels_y], dim=1)

            self.optimizer.zero_grad()
            flow_output = self.model(images)
            loss = self.criterion(flow_output, flow_truth)
            loss.backward()
            self.optimizer.step()

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
            for batch_idx, (images, flow_xs, flow_ys) in enumerate(self.val_loader):
                if self.cuda:
                    images = images.cuda()
                    flow_xs = flow_xs.cuda()
                    flow_ys = flow_ys.cuda()


                images = Variable(images)
                labels_x = Variable(flow_xs)
                labels_y = Variable(flow_ys)
                flow_truth = torch.cat([labels_x, labels_y], dim=1)
                flow_output = self.model(images)
                loss = self.criterion(flow_output, flow_truth)
                test_loss += loss.item()
            test_loss /= batch_idx
        self.txt_logger.info('\nTest set Epoch: {} Average loss: {:.4f}, \n'.format(epoch,  test_loss))
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
                        torch.save(self.model.module.cpu(), self.save_path + 'best_model.pt')
                    else:
                        torch.save(self.model.cpu(), self.save_path + 'best_model.pt')
                    self.model.cuda()
                else:
                    torch.save(self.model.cpu(), self.save_path + 'best_model.pt')


if __name__ == '__main__':
    model_config = Config(
        cuda=True if torch.cuda.is_available() else False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=2,
        lr=0.01,
        epochs=4,
        save_epoch=False,
        batch_size=16,
        log_interval=100,
        layers=[1, 1, 1, 1, 2],
        data_dir='dataset/processed/',
        save_path='output/'
    )

    trainer = FlowNetTrainer(model_config)
    print(trainer.model)
    # print(next(iter(trainer.train_loader)))
    print(next(trainer.model.encoder.parameters()).device)
    print(next(trainer.model.decoder.parameters()).device)


