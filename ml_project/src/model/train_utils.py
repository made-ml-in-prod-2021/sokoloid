#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import resnet18
from common_utils import GeoImage
import tqdm


class WalkLinesToDataset(data.Dataset):
    def __init__(self, image_file_name, walk_file_name, crop_size=16, walk_step=5, transforms=None):
        super(WalkLinesToDataset, self).__init__()
        self.transforms = transforms
        self.rectangle_size = crop_size
        self.map_image = GeoImage(image_file_name, crop_size)
        self.targets = []
        x_coords = []
        y_coords = []
        walks_df = pd.read_csv(walk_file_name, sep="\t", header=0)
        walks_df = walks_df.fillna(0).astype(np.int)
        for _, row in walks_df.iterrows():
            class_num = row[0]
            walk_points = row[1:].to_numpy().reshape(-1, 2)
            from_x, from_y = walk_points[0, 0], walk_points[0, 1]
            for to_x, to_y in walk_points[1:]:
                if to_x == 0 or to_y == 0:
                    break
                d_x = to_x - from_x
                d_y = to_y - from_y

                distance = (d_x ** 2 + d_y ** 2) ** 0.5
                steps = np.arange(0, distance, walk_step)
                size = steps.shape[0]

                x_steps = from_x + steps * d_x / distance
                y_steps = from_y + steps * d_y / distance
                self.targets.append(np.full((size,), class_num, dtype=np.int64))
                x_coords.append(x_steps.astype(np.int))
                y_coords.append(y_steps.astype(np.int))
                from_x, from_y = to_x, to_y

        self.targets = np.concatenate(self.targets)
        x_coords = np.concatenate(x_coords)
        y_coords = np.concatenate(y_coords)
        self.coords = np.stack([x_coords, y_coords], axis=1)

        assert len(self.targets) == self.coords.shape[0]

    def __getitem__(self, idx):
        sample = {"targets": self.targets[idx]}
        points = {"coord": self.coords[idx]}
        if self.transforms is not None:
            points = self.transforms(points)
        sample["image"] = self.map_image.get_rectangle(points["coord"],
                                             self.rectangle_size)

        sample["realcoord"] = points["coord"]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.targets)


class TransformByKeys:
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class RandomizeCoords:
    def __init__(self, deviation=5):
        self.deviation = deviation

    def __call__(self, coord):
        rand_offset = torch.randint(- self.deviation,
                                    self.deviation,
                                    (2,)).numpy()
        return coord + rand_offset


def train(model, loader, loss_fn, optimizer, scheduler,  device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        targets = batch["targets"]  # B x (2 * NUM_PTS)

        pred_targets = model(images).cpu()  # B x 2
        loss = loss_fn(pred_targets, targets)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
        images = batch["image"].to(device)
        targets = batch["targets"]

        with torch.no_grad():
            pred_targets = model(images).cpu()
        loss = loss_fn(pred_targets, targets)
        val_loss.append(loss.item())

    return np.mean(val_loss)


def make_model(num_labels: int):
    model = resnet18()
    classifier = nn.Sequential(nn.Linear(model.fc.in_features, 512),
                               nn.ReLU(),
                               nn.Linear(512, num_labels),
                               nn.LogSoftmax(dim=1))
    model.fc = classifier
    return model
