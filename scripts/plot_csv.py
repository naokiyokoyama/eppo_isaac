"""
This will plot reward curves using input csvs
"""
import argparse
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def convert_csv_to_df(csv_path):
    df = pd.read_csv(csv_path)
    method = osp.basename(csv_path).split("_")[0]
    seed = int(osp.basename(csv_path).split("seed_")[-1][0])
    df["method"] = [method] * df.shape[0]
    df["seed"] = [seed] * df.shape[0]
    return df


def main(csv_paths):
    all_frames = []
    for i in csv_paths:
        all_frames.append(convert_csv_to_df(i))
    df = pd.concat(all_frames, ignore_index=True)
    seaborn.lineplot(data=df, x="idx", y="rewards/step", hue="method")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="*")
    args = parser.parse_args()
    main(args.csv_paths)
