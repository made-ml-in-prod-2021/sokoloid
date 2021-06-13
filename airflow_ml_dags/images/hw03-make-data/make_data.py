import os
from sklearn.datasets import make_blobs

import click

X_FILENAME = "data.csv"
Y_FILENAME = "target.csv"


@click.command("make_data")
@click.option("--output_dir")
def make_data(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    X, y = make_blobs(n_samples=100, centers=3, n_features=2, )
    with open(os.path.join(output_dir, X_FILENAME), "wt") as f_out:
        for x_ in X:
            f_out.write(f"{x_[0]},{x_[1]}\n")

    with open(os.path.join(output_dir, Y_FILENAME), "wt") as f_out:
        for y_ in y:
            f_out.write(f"{y_}\n")

    os.makedirs(output_dir, exist_ok=True)


if __name__ == '__main__':
    make_data()
