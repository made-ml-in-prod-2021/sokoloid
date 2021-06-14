import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

import click

TRAIN_X_FILENAME = "train_data.csv"
TRAIN_Y_FILENAME = "train_target.csv"
MODEL_FILENAME = "model.pkl"



@click.command("validate")
@click.option("--input_dir")
@click.option("--output_dir")
def make_data(input_dir: str, output_dir: str):

    with open(os.path.join(input_dir, TRAIN_X_FILENAME), "rt") as f_in:
        x_data = np.loadtxt(f_in, delimiter=",")

    with open(os.path.join(input_dir, TRAIN_Y_FILENAME), "rt") as f_in:
        y_data = np.loadtxt(f_in, delimiter=",")
    clf = LogisticRegression().fit(x_data, y_data)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, MODEL_FILENAME), "wb") as f_out:
        pickle.dump(clf, f_out)

if __name__ == '__main__':
    make_data()
