import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import analysis_utils
import preprocess

if __name__ == "__main__":
    pos_band_value_dict = defaultdict(list)
    neg_band_value_dict = defaultdict(list)

    patch_path = "/run/media/aveekal/USB STICK/analytics/Very Large/patches/"

    for fire_id in os.listdir(patch_path):
        for date in os.listdir(os.path.join(patch_path, fire_id)):
            date_path = os.path.join(patch_path, fire_id, date)
            output_file = os.path.join(patch_path, fire_id, date, "output.tiff")
            img = np.array(Image.open(output_file))

            pos_pixels = np.where(img == 1)
            neg_pixels = np.where(img == 0)

            for file in os.listdir(date_path):
                if file == "output.tiff":
                    continue

                file_path = os.path.join(date_path, file)

                f = preprocess.parse_filename(file)
                img = np.array(Image.open(file_path))

                pos_band_value_dict[f["channel"]].append(img[pos_pixels].mean())
                neg_band_value_dict[f["channel"]].append(img[neg_pixels].mean())

    pos_band_value_dict = dict(
        map(
            lambda kv: (kv[0], analysis_utils.scale_values(kv[1], kv[0])),
            pos_band_value_dict.items(),
        )
    )
    neg_band_value_dict = dict(
        map(
            lambda kv: (kv[0], analysis_utils.scale_values(kv[1], kv[0])),
            neg_band_value_dict.items(),
        )
    )

    pos_mean_values = dict(
        map(lambda kv: (kv[0], sum(kv[1]) / len(kv[1])), pos_band_value_dict.items())
    )
    neg_mean_values = dict(
        map(lambda kv: (kv[0], sum(kv[1]) / len(kv[1])), neg_band_value_dict.items())
    )

    pos_mean_values = dict(sorted(pos_mean_values.items()))
    neg_mean_values = dict(sorted(neg_mean_values.items()))

    # Extracting keys and values
    keys = list(pos_mean_values.keys())
    pos_means = list(pos_mean_values.values())
    neg_means = list(neg_mean_values.values())

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = range(len(keys))
    r2 = [x + bar_width for x in r1]

    # Create the grouped bar plot
    plt.bar(
        r1,
        pos_means,
        color="r",
        width=bar_width,
        edgecolor="grey",
        label="Fire Pixel",
    )

    plt.bar(
        r2,
        neg_means,
        color="b",
        width=bar_width,
        edgecolor="grey",
        label="Non-Fire Pixel",
    )

    # Add labels, title, and legend
    plt.xlabel("Band")
    plt.ylabel("Mean Values")
    plt.title("Mean Values for Fire and Non-Fire Pixels")
    plt.xticks([r + bar_width / 2 for r in range(len(keys))], keys)
    plt.legend()

    # Save the plot
    # plt.savefig(os.path.join(patch_path, ))
    plt.savefig(f"{patch_path.split('/')[-3]}.png")
