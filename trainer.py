import os
import logging
import cv2
import torch
import wandb

from tqdm.auto import tqdm
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
        self.loss_type = cfg.loss_type
        self.device = cfg.device

        # stuff
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD

        # other stuff
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
        # set logging level
        logging.basicConfig(filename='train.log', level=logging.INFO)

        # initialize WandbLogger
        self.wandb_logger = WandbLogger(project=self.cfg.project)

        # initialize a set of noise vectors which will be used to
        # visualize generator's progress
        self.noise = generate_noise(16, self.z_dim).to(self.device)
        os.makedirs(self.cfg.save_folder, exist_ok=True)

        #TODO: log config

    def after_train(self):
        # finish wandb logger
        self.wandb_logger.finish()

        # save current models
        save_checkpoint(self.discriminator,
                        save_dir='weights/',
                        model_name='discriminator')
        save_checkpoint(self.generator,
                        save_dir='weights/',
                        model_name='generator')

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
        # save to disk
        for i in range(ims.size()[0]):
            im = ims[i].permute(1,2,0)
            im = im * 255
            im = im.cpu().detach().numpy()
            cv2.imwrite(os.path.join(self.cfg.save_folder, f"{self.epoch}_{i}.jpg"), im)

        ims = [wandb.Image(ims[i]) for i in range(ims.size()[0])]
        values = {"Generator Progress": ims}

        self.wandb_logger.log_metrics(values)
        logging.info(f"Epoch {self.epoch}   D_Loss: {self.D_loss.val}   G_loss: {self.G_loss.val}"
                     f"D_G_z: {self.D_G_z.val}  D_x: {self.D_x.val}")

    def train_in_iter(self):
        with tqdm(range(self.cfg.max_iter), position=0, leave=True,
                  desc=f"Epoch {self.epoch}") as t:
            for self.iter in t:
                self.before_iter()
                self.train_one_iter()
                self.after_iter(t)

    def before_iter(self):
        pass

    def after_iter(self, t):
        """
        1. log metrics to WandB
        2. print progress
        """
        values = {"D_loss": self.D_loss.value,
                  "G_loss": self.G_loss.value,
                  "D(G(z))": self.D_G_z.value,
                  "D(x)": self.D_x.value,
                 }
        self.wandb_logger.log_metrics(metrics=values)

        #TODO: print this without writing to new line
        #t.set_description('Epoch {} [{}/{}] '
        #                  # 'time: {:.3f}\n'
        #                  'D_loss: {D_loss.value:.3f} ({D_loss.val:.3f})\n'
        #                  'G_loss: {G_loss.value:.3f} ({G_loss.val:.3f})\n'
        #                  'D(G(z)): {D_G_z.value:.3f} ({D_G_z.val:.3f})\n'
        #                  'D(x): {D_x.value:.3f} ({D_x.val:.3f})\n'
        #                  .format(self.epoch,
        #                          self.iter + 1,
        #                          self.max_iter,
        #                          # batch_time,
        #                          D_loss=self.D_loss,
        #                          G_loss=self.G_loss,
        #                          D_G_z=self.D_G_z,
        #                          D_x=self.D_x,
        #                          ),
        #                  refresh=True)
        #t.clear()

    def train_one_iter(self):
        for i in range(self.cfg.nD):
            images = next(self.dataloader_iterator).to(self.device)
            self.train_one_iter_D(images)
        for i in range(self.cfg.nG):
            images = None
            if self.loss_type in ['feature_matching', 'both']:
                images = next(self.dataloader_iterator).to(self.device)
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


