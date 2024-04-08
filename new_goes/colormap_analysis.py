import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import analysis_utils
import preprocess

GRID_WIDTH = 16
GRID_HEIGHT = 1
OFFSET = 1
IMG_WIDTH = 32
TOTAL_WIDTH = (IMG_WIDTH + OFFSET) * GRID_WIDTH - OFFSET
cm = plt.get_cmap("viridis")


def band_file_dict(date_path: str):
    dict = {}

    for file in os.listdir(date_path):
        if file == "output.tiff":
            continue

        f = preprocess.parse_filename(file)
        dict[f["channel"]] = file

    return dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()

    save_dir = "colormap"
    fire_type = ""

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    patch_path = args.path

    for fire_id in os.listdir(patch_path):
        fire_id_path = os.path.join(save_dir, fire_id)
        fire_id_path = os.listdir(os.path.join(patch_path, fire_id))

        final_image = Image.new(
            "RGB",
            (TOTAL_WIDTH, (IMG_WIDTH + OFFSET) * len(fire_id_path) - OFFSET),
            color="black",
        )

        for x, date in enumerate(sorted(fire_id_path)):
            result_image = Image.new("RGB", (TOTAL_WIDTH, IMG_WIDTH), color="black")
            date_path = os.path.join(patch_path, fire_id, date)

            dict = band_file_dict(date_path)

            for i in range(GRID_HEIGHT):
                for j in range(GRID_WIDTH):
                    img_band = dict[i * GRID_WIDTH + j + 1]
                    img_path = os.path.join(date_path, img_band)

                    f = preprocess.parse_filename(img_path.split("/")[-1])

                    img = Image.open(img_path)
                    im = np.array(img).tolist()
                    im = analysis_utils.scale_values(im, f["channel"])
                    im = np.uint8(cm(im) * 255)
                    im = Image.fromarray(im)

                    result_image.paste(
                        im, (j * (IMG_WIDTH + OFFSET), i * (IMG_WIDTH + OFFSET))
                    )

            final_image.paste(result_image, (0, x * (IMG_WIDTH + OFFSET)))

        final_image.save(f"{save_dir}/{fire_id}.png")