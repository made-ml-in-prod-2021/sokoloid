import os
import numpy as np
from sklearn.model_selection import train_test_split

import click

X_FILENAME = "data.csv"
Y_FILENAME = "target.csv"

TRAIN_X_FILENAME = "train_data.csv"
TRAIN_Y_FILENAME = "train_target.csv"
TEST_X_FILENAME = "test_data.csv"
TEST_Y_FILENAME = "test_target.csv"
SPLIT_RATIO = 0.8


@click.command("split_data")
@click.option("--input_dir")
@click.option("--output_dir")
def make_data(input_dir: str, output_dir: str):
    with open(os.path.join(input_dir, X_FILENAME), "rt") as f_in:
        x_data = np.loadtxt(f_in, delimiter=",")

    with open(os.path.join(input_dir, Y_FILENAME), "rt") as f_in:
        y_data = np.loadtxt(f_in, delimiter=",")

    arrays = train_test_split(x_data, y_data, train_size=SPLIT_RATIO)

    file_names = [TRAIN_X_FILENAME,
                  TEST_X_FILENAME,
                  TRAIN_Y_FILENAME,
                  TEST_Y_FILENAME]

    for file_name, array in zip(file_names, arrays):
        with open(os.path.join(output_dir, file_name), "wt") as f_out:
            np.savetxt(f_out, array, delimiter=",")


if __name__ == '__main__':
    make_data()
