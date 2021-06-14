import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle

import click

DATA_FILENAME = "data.csv"
MODEL_FILENAME = "model.pkl"
PREDICT_FILENAME = "predictions.csv"


@click.command("train")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--model_dir")
def make_data(input_dir: str, model_dir: str, output_dir:str):
    with open(os.path.join(input_dir, DATA_FILENAME), "rt") as f_in:
        data = np.loadtxt(f_in, delimiter=",")

    with open(os.path.join(model_dir, MODEL_FILENAME), "rb") as f_in:
        clf = pickle.load(f_in)

    predict = clf.predict(data)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, PREDICT_FILENAME), "wt") as f_out:
        np.savetxt(f_out, predict, delimiter=",")


if __name__ == '__main__':
    make_data()
