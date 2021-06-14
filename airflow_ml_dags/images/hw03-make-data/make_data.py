import os
from sklearn.datasets import make_blobs

import click

FEATURES_NUM = 2
CLASS_NUM = 3
DATA_LEN = 1000

X_FILENAME = "data.csv"
Y_FILENAME = "target.csv"


@click.command("make_data")
@click.option("--output_dir")
def make_data(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    X, y = make_blobs(n_samples=DATA_LEN, centers=CLASS_NUM, n_features=FEATURES_NUM, )
    print(f"write data to {output_dir}")
    with open(os.path.join(output_dir, X_FILENAME), "wt") as f_out:
        for x_ in X:
            f_out.write(f"{x_[0]},{x_[1]}\n")

    with open(os.path.join(output_dir, Y_FILENAME), "wt") as f_out:
        for y_ in y:
            f_out.write(f"{y_}\n")


if __name__ == '__main__':
    make_data()
