#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
общие утилиты
"""
import os


def norm_file_path(file_path, norm_path):
    if not os.path.isabs(os.path.dirname(file_path)):
        file_path = os.path.join(norm_path, file_path)
    return file_path
