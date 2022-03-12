import time
import tqdm
import torch
import logging
import wandb

from torch import nn

# Custom module
from utils import AverageMeter, generate_noise, WandbLogger, \
    save_checkpoint

class Trainer:
    def __init__(self,
                 cfg,
                 generator,
                 discriminator,
                 dataloader,
                 optimizerG,
                 optimizerD,
                 ):
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        self.max_epoch = cfg.max_epoch
        self.batch_size = cfg.batch_size
        self.device = cfg.device

        # stuff
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD

        # other stuff
        #self.loss_type = "adversarial"
        self.loss_type = "feature_matching"
        self.dataloader_iterator = iter(self.dataloader)

        # Loss Functions
        self.adv_loss = nn.BCELoss()
        self.fm_loss = nn.MSELoss(reduction='mean')  # feature matching


    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        # initialize WandbLogger
        logging.basicConfig(filename='train.log', level=logging.DEBUG)
        self.wandb_logger = WandbLogger(project=self.cfg.project)

        # set max iteration per epoch
        num_samples = self.cfg.n_samples_per_epoch
        one_iter_batch_size = (self.cfg.nG + self.cfg.nD) * self.batch_size
        self.max_iter = num_samples // one_iter_batch_size

        logging.info(f"max_iter: {self.max_iter}")

        # initialize a set of noise vectors which will be used to
        # visualize generator's progress
        self.noise = generate_noise(16, self.z_dim)

    def after_train(self):
        # finish wandb logger
        self.wandb_logger.finish()

        # save current models
        save_checkpoint(self.discriminator,
                        save_dir='weights/',
                        model_name='Discriminator')
        save_checkpoint(self.generator,
                        save_dir='weights/',
                        model_name='Generator')

    def train_in_epoch(self):
        for self.epoch in range(self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def before_epoch(self):
        self.D_loss = AverageMeter()
        self.G_loss = AverageMeter()
        self.D_G_z = AverageMeter()
        self.D_x = AverageMeter()
        self.D_acc = AverageMeter()

    def after_epoch(self):
        # log images to wandb
        #TODO: implement this as a method of wandblogger class
        ims = self.generator(self.noise)
        ims = (ims + 1) / 2
        ims = [wandb.Image(ims[i]) for i in range(ims.size()[0])]
        values = {"Generator Progress": ims}

        self.wandb_logger.log_metrics(values)
        logging.info(f"Epoch {self.epoch}   D_Loss: {self.D_loss.val}   G_loss: {self.G_loss.val}"
                     f"D_G_z: {self.D_G_z.val}  D_x: {self.D_x.val}")

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def before_iter(self):
        pass

    def after_iter(self):
        values = {"D_loss": self.D_loss.value,
                  "G_loss": self.G_loss.value,
                  "D(G(z))": self.D_G_z.value,
                  "D(x)": self.D_x.value,
                 }

        self.wandb_logger.log_metrics(metrics=values)

    def train_one_iter(self):
        for i in range(self.cfg.nD):
            images = next(self.dataloader_iterator)
            self.train_one_iter_D(images)
        for i in range(self.cfg.nG):
            images = None
            if self.loss_type in ['feature_matching', 'both']:
                images = next(self.dataloader_iterator)
            self.train_one_iter_G(images)

    def train_one_iter_D(self, images):
        self.optimizerD.zero_grad()
        # get noise (z)
        noise = generate_noise(self.batch_size, self.z_dim, type='normal')
        noise = noise.to(self.device)
        # get fake and real images
        real_label = torch.ones(self.batch_size).to(self.device)
        fake_label = torch.zeros(self.batch_size).to(self.device)

        # forward and backward
        fake_images = self.generator(noise)
        fake_pred, _ = self.discriminator(fake_images)
        fake_loss = self.adv_loss(fake_pred, fake_label)

        real_pred, _ = self.discriminator(images)
        real_loss = self.adv_loss(real_pred, real_label)

        # accumulate gradient and update
        D_loss = fake_loss + real_loss
        D_loss.backward()
        self.optimizerD.step()

        # logging
        D_x = real_pred.mean(0).item()
        D_G_z = fake_pred.mean(0).item()
        self.D_x.update(D_x, self.batch_size)
        self.D_G_z.update(D_G_z, self.batch_size)
        self.D_loss.update(D_loss, self.batch_size)

    def train_one_iter_G(self, images=None):
        self.optimizerG.zero_grad()
        # get noise (z)
        noise = generate_noise(self.batch_size, self.z_dim, type='normal')
        noise = noise.to(self.device)
        # get real labels ( modified loss, minimize -log(D(G(z)) )
        real_label = torch.ones(self.batch_size).to(self.device)
        fake_images = self.generator(noise)

        if self.loss_type == 'adversarial':
            # vanilla adversarial loss
            fake_pred, _ = self.discriminator(fake_images)
            G_loss = self.adv_loss(fake_pred, real_label)
            G_loss.backward()

        elif self.loss_type == 'feature_matching':
            _, fmaps_fake = self.discriminator(fake_images)
            _, fmaps_real = self.discriminator(images)
            # get feature map statistics of last layer
            fmap_fake = fmaps_fake[-1].mean(0)
            fmap_real = fmaps_real[-1].mean(0).detach()  # treat as constant
            G_loss = self.fm_loss(fmap_fake, fmap_real)
            G_loss.backward()

        elif self.loss_type == 'both':
            raise ValueError("los_type = both is not yet implemented")

        else:
            raise ValueError("invalid loss_type.")

        self.optimizerG.step()
        self.G_loss.update(G_loss, self.batch_size)


