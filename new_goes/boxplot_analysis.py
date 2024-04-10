import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import analysis_utils
import preprocess

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()

    pos_band_value_dict = defaultdict(list)
    neg_band_value_dict = defaultdict(list)

    patch_path = args.path

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

                pos_band_value_dict[f["channel"]].extend(img[pos_pixels])
                neg_band_value_dict[f["channel"]].extend(img[neg_pixels])

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


    pos_mean_values = dict(sorted(pos_band_value_dict.items()))
    neg_mean_values = dict(sorted(neg_band_value_dict.items()))

    keys = list(pos_mean_values.keys())
    pos_values = list(pos_mean_values.values())
    neg_values = list(neg_mean_values.values())

    plt.figure()
    fire = plt.boxplot(pos_values, positions=np.arange(len(pos_values))*3, sym='', widths=0.6)
    non_fire = plt.boxplot(neg_values, positions=np.arange(len(neg_values))*3+1, sym='', widths=0.6)

    set_box_color(fire, '#D7191C')
    set_box_color(non_fire, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Fire')
    plt.plot([], c='#2C7BB6', label='Non_Fire')
    plt.legend()

    plt.xticks(np.arange(len(keys))*3+0.5, keys)
    plt.tight_layout()
    plt.savefig(f"{patch_path.split('/')[-3]}.png")

