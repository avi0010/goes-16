import json
from typing import List
from PIL import Image
from osgeo import osr, gdal
import os
from tqdm import tqdm
import uuid
from dotenv import load_dotenv
load_dotenv()

class Node:
    def __init__(self) -> None:
        base_dir = os.getenv("BASE_DATA_DIR")
        if base_dir is None:
            raise ValueError("BASE_DATA_DIR is not provided in env file")

        self.base_dir = base_dir

        if len(os.listdir(self.base_dir)) % 16 != 0 :
            raise ValueError("Base files not present")

        patches_dir = os.getenv("BASE_PATCHES_DIR")
        if patches_dir is None:
            raise ValueError("BASE_PATCHES_DIR is not provided in env file")

        self.patches_dir = patches_dir
        if not os.path.exists(self.patches_dir):
            os.mkdir(self.patches_dir)

    def process(self):
        for file in os.listdir(self.base_dir):
            file_path = os.path.join(self.base_dir, file)
            dataset = gdal.Open(file_path)

            for i in tqdm(range(0, 9984, 32)):
                for j in range(0, 5984, 32):
                    patch = dataset.ReadAsArray(i, j, 32, 32)

                    patch_path = os.path.join(self.patches_dir, f"{i}_{j}")
                    if not os.path.exists(patch_path):
                        os.mkdir(patch_path)

                    # Create a new dataset for the patch
                    driver = gdal.GetDriverByName('GTiff')
                    patch_file_name = os.path.join(patch_path, file)
                    out_dataset = driver.Create(patch_file_name, 32, 32, dataset.RasterCount, gdal.GDT_Byte)

                    # Write the patch data to the new dataset
                    out_dataset.GetRasterBand(1).WriteArray(patch)

                    # Set the geo-transform and projection if necessary
                    geotransform = list(dataset.GetGeoTransform())
                    geotransform[0] += i * geotransform[1]  # Adjust the x origin
                    geotransform[3] += j * geotransform[5]  # Adjust the y origin
                    out_dataset.SetGeoTransform(geotransform)
                    out_dataset.SetProjection(dataset.GetProjection())

                    out_dataset = None

            dataset = None

if __name__ == '__main__':
    Node().process()
