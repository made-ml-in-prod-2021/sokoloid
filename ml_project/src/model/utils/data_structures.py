#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
набор классов для конфигурации приложения


"""
from dataclasses import dataclass
from omegaconf import DictConfig, MISSING
from PIL import Image, ImageDraw
import numpy as np
import logging

log = logging.getLogger(__name__)


class GeoImage:
    def __init__(self, file_name: str, crop_size: int):
        self.crop_size = crop_size
        self.image = None
        self.load_image(file_name)

    def load_image(self, file_name: str):
        log.info(f"Try to load map image {file_name}")
        self.image = Image.open(file_name)

    def get_rectangle(self, coord: np.array, size: int = -1):
        if size == -1:
            size = self.crop_size
        half_size = size // 2
        rect = np.concatenate([coord - half_size, coord + half_size]).astype(int)
        img = self.image.crop(rect)
        return img.convert('RGB')

    def save_debug_image(self, file_name: str,
                         coord: np.array,
                         crop_size: int,
                         analyzed_area=None,
                         ):

        im_crop = self.get_rectangle(coord, crop_size)

        draw = ImageDraw.Draw(im_crop)

        half_size = crop_size // 2
        half_crop = analyzed_area // 2
        if analyzed_area is not None:
            draw.line((half_size - half_crop, half_size - half_crop,
                       half_size + half_crop, half_size - half_crop))
            draw.line((half_size - half_crop, half_size + half_crop,
                       half_size + half_crop, half_size + half_crop))
            draw.line((half_size - half_crop, half_size - half_crop,
                       half_size - half_crop, half_size + half_crop))
            draw.line((half_size + half_crop, half_size - half_crop,
                       half_size + half_crop, half_size + half_crop))

        # рисуем крест
        draw.line((half_size, 0, half_size, crop_size))
        draw.line((0, half_size, crop_size, half_size))

        im_crop.save(file_name)

@dataclass
class MapConfig(DictConfig):
    map_image: str = MISSING
    model_pkl: str = MISSING
    max_image_pixels: int = MISSING
    gps_coord: list = MISSING
    pixel_coord: list = MISSING
    crop_size: int = MISSING
    class_list: dict = MISSING


@dataclass
class Config(DictConfig):
    input: str = "input.csv"
    output: str = "output.csv"
    debug: bool = False
    debug_crop_size: int = 128
    debug_image_dir: str = "c:/temp"
    batch_size: int = 512
    threshold: float = 0.95
    # путь к корневому каталогу ПРОГРАММЫ
    path_to_root: str = MISSING
    map: MapConfig = MISSING


@dataclass
class PredictResult:
    coord: tuple = (0, 0)
    target: int = 0
    predict: int = 0
    score: float = 0