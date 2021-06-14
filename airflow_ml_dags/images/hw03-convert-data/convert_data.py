import os
import numpy as np
from sklearn.model_selection import train_test_split

import click

X_FILENAME = "data.csv"
Y_FILENAME = "target.csv"


SPLIT_RATIO = 0.8


@click.command("convert_data")
@click.option("--input_dir")
@click.option("--output_dir")
def make_data(input_dir: str, output_dir: str):

    data  = []
    files = [X_FILENAME, Y_FILENAME]
    for file_name in files:
        with open(os.path.join(input_dir, file_name), "rt") as f_in:
            data.append(np.loadtxt(f_in, delimiter=","))

    os.makedirs(output_dir, exist_ok=True)
    for file_name, array in zip(files, data):
        with open(os.path.join(output_dir, file_name), "wt") as f_out:
            np.savetxt(f_out, array, delimiter=",")


if __name__ == '__main__':
    make_data()
