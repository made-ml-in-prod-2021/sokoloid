#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
классы для работы с данными
"""
import os
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


class GpsToPixelTransformer:
    '''
    трансформер для преобразования коррдинат GPS в пиксели

    '''

    def __init__(self, gps_coord=None, pixels_coord=None):
        # матрица линейных преобразований координат
        self.rotate_matrix = None
        if gps_coord is not None and pixels_coord is not None:
            self.fit(gps_coord, pixels_coord)

    def fit(self, gps_coord, pixels_coord):
        """
        формирование матрицы преобразований
        :param pixels_coord: координанты 3х точек pixels numpy.array  shape (3, 2)
        :param gps_coord: координанты 3х точек GPS numpy.array  shape (3, 2)   dtype=np.float64

        :return:
        """

        gps = np.ones((3, 3), dtype=np.float64)
        gps[:, [0, 1]] = gps_coord

        pixels = np.ones((3, 3), dtype=np.float64)
        pixels[:, [0, 1]] = pixels_coord

        self.rotate_matrix = np.linalg.inv(gps) @ pixels

    def transform(self, gps_coord: np.array):
        """
        преобразование координат. на входе  GPS numpy.array  shape ( N, 2)
        :param gps_coord: GPS numpy.array  shape (2,)

        :return:  pixels numpy.array  shape (2,)
        """

        coord = np.ones((3,), dtype=np.float64)

        coord[[0, 1]] = gps_coord
        transformed = coord.reshape((1, 3)) @ self.rotate_matrix
        return transformed[0, [0, 1]].astype(np.int)


def norm_file_path(file_path, norm_path):
    if not os.path.isabs(os.path.dirname(file_path)):
        file_path = os.path.join(norm_path, file_path)
    return file_path
