import os
import torch

from torch.utils.data import DataLoader, RandomSampler

# Custom Modules
from utils import weights_init
from model import Generator, Discriminator
from dataset import CustomDataset
from trainer import Trainer

class CFG:
    def __init__(self):
        self.max_epoch = 10
        self.batch_size = 8
        self.z_dim = 64
        self.n_feat = 64
        self.folder = './data/img_align_celeba/img_align_celeba/'
        self.dataset = CustomDataset(self.folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    cfg = CFG()
    ####### Define data loader and models ##########
    train_sampler = RandomSampler(data_source=cfg.dataset,
                                  replacement=True,
                                  num_samples=9000,
                                  )
    dataloader = DataLoader(cfg.dataset,
                            batch_size=cfg.batch_size,
                            sampler=train_sampler,
                            pin_memory=True,
                            #shuffle=True,
                            num_workers=4,
                            )
    G = Generator(z_dim=cfg.z_dim,
                  n_feat=cfg.n_feat,
                  )

    D = Discriminator(n_feat=cfg.n_feat)

    G.apply(weights_init)
    D.apply(weights_init)

    G = G.to(cfg.device)
    D = D.to(cfg.device)

    params_G = G.parameters()
    params_D = D.parameters()

    optimizerG = torch.optim.Adam(params=params_G,
                            lr=2e-4,
                            betas=(0.5, 0.999),
                            weight_decay=1e-3,
                            )
    optimizerD = torch.optim.Adam(params=params_D,
                            lr=2e-4,
                            betas=(0.5, 0.999),
                            weight_decay=1e-3,
                            )

    cfg.dataloader = dataloader
    trainer = Trainer(cfg,
                      G,
                      D,
                      dataloader,
                      optimizerG,
                      optimizerD,
                      )

    trainer.train()
