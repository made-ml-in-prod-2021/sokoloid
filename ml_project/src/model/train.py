#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1

"""

import os.path
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import hydra
from train_utils import WalkLinesToDataset, TransformByKeys, RandomizeCoords, train, validate, make_model

log = logging.getLogger(__name__)

@hydra.main(config_path="conf",
            config_name='train_config.yaml')
def main(cfg: DictConfig) -> None:
    log.info("Start training")
    norm_path = os.path.normpath(os.path.join(os.getcwd(), cfg.path_to_root))
    Image.MAX_IMAGE_PIXELS = cfg.map.max_image_pixels
    train_transforms = transforms.Compose([
        TransformByKeys(RandomizeCoords(deviation=cfg.deviation), ("coord",)),
        TransformByKeys(transforms.RandomHorizontalFlip(), ("image",)),
        TransformByKeys(transforms.RandomVerticalFlip(), ("image",)),

        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])
    walk_lines = WalkLinesToDataset(walk_file_name=os.path.join(norm_path, cfg.walks_file),
                                    image_file_name=os.path.join(norm_path, cfg.map.map_image),
                                    transforms=train_transforms,
                                    crop_size=cfg.crop_size,
                                    walk_step=cfg.walk_step)
    dataset_len = len(walk_lines)
    log.info(f"Dataset len {dataset_len}")
    train_len = int(dataset_len * cfg.train_val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(walk_lines, [train_len, dataset_len - train_len])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Working on {device}")

    model = make_model(len(cfg.map.class_list))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, )
    loss_fn = nn.NLLLoss().to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                              steps_per_epoch=int(len(train_data_loader)),
                                              epochs=cfg.epoch,
                                              anneal_strategy='linear')

    # 2. train & validate
    log.info("Ready for training...")
    best_val_loss = np.inf

    for epoch in range(cfg.epoch):
        train_loss = train(model, train_data_loader, loss_fn, optimizer, scheduler, device=device)
        val_loss = validate(model, val_data_loader, loss_fn, device=device)
        log.info("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(norm_path, cfg.model_save))

    log.info(f"best val is  {best_val_loss}")
    return 0


if __name__ == "__main__":
    main()
