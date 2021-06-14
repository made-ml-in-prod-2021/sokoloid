import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle

import click

TEST_X_FILENAME = "test_data.csv"
TEST_Y_FILENAME = "test_target.csv"

MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.csv"


@click.command("train")
@click.option("--input_dir")
@click.option("--model_dir")
def make_data(input_dir: str, model_dir: str):
    with open(os.path.join(input_dir, TEST_X_FILENAME), "rt") as f_in:
        x_data = np.loadtxt(f_in, delimiter=",")

    with open(os.path.join(input_dir, TEST_Y_FILENAME), "rt") as f_in:
        y_data = np.loadtxt(f_in, delimiter=",")

    with open(os.path.join(model_dir, MODEL_FILENAME), "rb") as f_in:
        clf = pickle.load(f_in)

    predict = clf.predict(x_data)
    result = np.array(precision_recall_fscore_support(y_data, predict))

    with open(os.path.join(model_dir, METRICS_FILENAME), "wt") as f_out:
        f_out.write(str(result))


if __name__ == '__main__':
    make_data()
