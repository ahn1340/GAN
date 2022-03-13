import os
import torch
import ruamel.yaml as yaml
import logging

from attrdict import AttrDict
from torch.utils.data import DataLoader, RandomSampler

# Custom Modules
from utils import weights_init, visualize_progress
from model import Generator, Discriminator
from dataset import CustomDataset
from trainer import Trainer

if __name__ == "__main__":
    # load configs
    with open("config/hyperparameters.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = AttrDict(cfg)

    # set device
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # create dataset
    dataset = CustomDataset(os.path.abspath(cfg.folder))

    ####### Define data loader and models ##########
    train_sampler = RandomSampler(data_source=dataset,
                                  replacement=True,
                                  num_samples=int(1e100),  # make the dataloader "infinite"
                                  )
    dataloader = DataLoader(dataset,
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
                            lr=cfg.lr_G,
                            betas=(cfg.beta1_G, cfg.beta2_G),
                            weight_decay=cfg.weight_decay_G,
                            )
    optimizerD = torch.optim.Adam(params=params_D,
                            lr=cfg.lr_D,
                            betas=(cfg.beta1_D, cfg.beta2_D),
                            weight_decay=cfg.weight_decay_D,
                            )

    trainer = Trainer(cfg,
                      G,
                      D,
                      dataloader,
                      optimizerG,
                      optimizerD,
                      )

    # Train models
    trainer.train()

    # create gif of progress
    visualize_progress(cfg.save_folder, cfg.max_epoch)
