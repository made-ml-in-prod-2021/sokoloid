#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 6.0.1
программа анализа геоданных по изображению генплана города.
на входе файл массив с GPS координатами в формате
 [55.86605046256052, 49.07751363142922]
[55.85844989210669, 49.097790215192205]
[55.86076205652522, 49.1011128167036]
одна строка - 1 запрос
 на выходе файл с описанием

[55.79883508185922, 49.105875912272566];14404;12460;Специализированная зона размещения объектов торговли...
[55.80330934948255, 49.33272207144011];25250;12114;Зона мест погребения


"""

import os.path
import logging
import functools
import operator
import numpy as np
import torch
from PIL import Image
import hydra
import tqdm

from ml_project.src.model.utils.common_utils import GeoImage, norm_file_path
from ml_project.src.model.utils.transformers import GeoTransformer, GpsToPixelTransformer
from ml_project.src.model.utils.datasets import QueryDataset
from ml_project.src.model.utils.data_structures import Config

log = logging.getLogger(__name__)


def predict_area_class(model, data_loader, cfg):
    gps = []
    pixels = []
    predicts = []
    log.info(f"Ready to predict for {len(data_loader)} batches")
    for batch in tqdm.tqdm(data_loader,
                           total=len(data_loader), desc="recognizing...",
                           position=0, unit="points", unit_scale=cfg.batch_size,
                           leave=True):
        with torch.no_grad():
            predicts.append(model(batch["tensor"]).cpu().numpy())
            gps.append(batch["gps"])
            pixels.append(batch["pixels"].numpy())
    gps = functools.reduce(operator.iconcat, gps, [])
    pixels = np.concatenate(pixels, axis=0)
    predicts = np.concatenate(predicts, axis=0)
    log.info(f"  {len(gps)} records processed")
    return gps, pixels, predicts


def write_predict_w_description(output_file_name: str,
                                prediction_results: tuple,
                                cfg,
                                geo_map):
    gps_list, pixels_arr, predicts_arr = prediction_results
    sorted_class_ids_list = np.argsort(predicts_arr, axis=1)
    class_dict = {int(k): cfg.map.class_list[k] for k in cfg.map.class_list}
    log.info(f" Try to save to   {output_file_name} file")
    cnt = 0
    with open(output_file_name, "wt") as fout:
        for gps, pixels, sorted_class_ids, predict in zip(gps_list,
                                                          pixels_arr,
                                                          sorted_class_ids_list,
                                                          predicts_arr):
            prob = np.exp(-predict[sorted_class_ids[-1]])
            predicted_class1 = sorted_class_ids[-1]
            predicted_class2 = sorted_class_ids[-2]
            if prob > cfg.threshold:
                description = f"{class_dict[predicted_class1]}"
            else:
                description = f"{class_dict[predicted_class1]}_or_{class_dict[predicted_class2]}"

            fout.write(f"{gps};{pixels[0]};{pixels[1]};{description}\n")
            cnt += 1
            if cfg.debug:
                file_name = f"img_{gps.replace('.', '_')}" \
                            f"_x{pixels[0]}-y{pixels[1]}" \
                            f"_{round(100 * prob)}_" \
                            f"_{predicted_class1}" \
                            f"_{predicted_class2}_.jpg"
                file_name = os.path.join(cfg.debug_image_dir, file_name)
                geo_map.save_debug_image(file_name,
                                         pixels,
                                         crop_size=cfg.debug_crop_size,
                                         analyzed_area=cfg.map.crop_size)
        log.info(f" saved {cnt} records")


@hydra.main(config_path="conf",
            config_name="config.yaml")
def main(cfg: Config) -> None:
    log.info(f"Start")
    root_path = os.path.normpath(os.path.join(hydra.utils.get_original_cwd(), cfg.path_to_root))
    input_file = norm_file_path(cfg.input, root_path)
    output_file = norm_file_path(cfg.output, root_path)

    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, cfg.map.max_image_pixels)
    geo_map = GeoImage(os.path.join(root_path, cfg.map.map_image), cfg.map.crop_size)

    gps_to_pixel = GpsToPixelTransformer(np.array(cfg.map.gps_coord),
                                         np.array(cfg.map.pixel_coord))

    to_tensor_transformer = GeoTransformer(geo_map)
    log.info(f"Loading  model {cfg.map.model_pkl}")
    model = torch.load(os.path.join(root_path, cfg.map.model_pkl))
    model.eval()

    query_dataset = QueryDataset(input_file,
                                 gps_to_pixel.transform,
                                 to_tensor_transformer
                                 )
    query_data_loader = torch.utils.data.DataLoader(query_dataset,
                                                    batch_size=cfg.batch_size,
                                                    # num_workers=1,
                                                    shuffle=False,
                                                    drop_last=False)

    prediction_results = predict_area_class(model, query_data_loader, cfg)

    write_predict_w_description(output_file, prediction_results, cfg, geo_map)
    log.info("End processing data")


if __name__ == "__main__":
    main()
