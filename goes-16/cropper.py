from collections import defaultdict
import numpy as np
from osgeo import ogr, gdal, osr
from PIL import Image
from datetime import datetime, timedelta
from bbox import Bboxs, Bbox
import s3fs
import random
import os
import json
from netCDF4 import Dataset
import argparse
from tqdm import tqdm

def get_center_pixel(label_path, mask_value:int = 1):
    ds = gdal.Open(label_path)
    band = ds.GetRasterBand(1)
    array = np.array(band.ReadAsArray())

    # Find indices where we have mass
    mass_x, mass_y = np.where(array == mask_value)

    # mass_x and mass_y are the list of x indices and y indices of mass pixels  
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)

    # Clean up
    ds = None

    return int(cent_x), int(cent_y)

def label_generator(raster_path, save_loc='/app/VALID', default_confidence_value=0):

    conf_map = {10: 1.0,
                30: 1.0,
                11: 0.9,
                31: 0.9,
                12: 0.8,
                32: 0.8,
                13: 0.5,
                33: 0.5,
                14: 0.3,
                34: 0.3,
                15: 0.1,
                35: 0.1,
                }

    inDs = gdal.Open(f"NETCDF:{raster_path}:{'Mask'}", gdal.GA_ReadOnly)
    band1 = inDs.GetRasterBand(1)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    cropData = band1.ReadAsArray(0,0,cols,rows)

    driver = inDs.GetDriver()
    outDs = driver.Create(f"{save_loc}/mask.tif", cols, rows, 1, gdal.GDT_Float32)
    outBand = outDs.GetRasterBand(1)
    outData = np.ones(cropData.shape, dtype=np.float32) * default_confidence_value
    for v, conf in conf_map.items():
        outData[cropData == v] = conf
    
    fire_indexes = np.where(outData > 0)
    print(fire_indexes)

    outBand.WriteArray(outData)
    outBand.FlushCache()
    outBand.SetNoDataValue(default_confidence_value)
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    del outData
    inDs = None

def crop(reference_raster_path, save_loc='/app/VALID'):

    mask_file = f"{save_loc}/mask.tif"
    centre_x, centre_y = get_center_pixel(mask_file)

    win_size = 32
    # Finding offset by moving the center pixels to top-left according to image size
    x_random = int(random.uniform(-1 * (win_size // 3), win_size // 3))
    y_random = int(random.uniform(-1 * (win_size // 3), win_size // 3))
    x_offset = centre_y - win_size // 2 + x_random
    y_offset = centre_x - win_size // 2 + y_random

    window = (x_offset, y_offset, win_size, win_size)
    # Translate output.tif
    gdal.Translate(f"{save_loc}/cropped_mask.tif", mask_file, srcWin=window)

    # Translate reference raster
    inDs = gdal.Open(f"NETCDF:{reference_raster_path}:{'Mask'}", gdal.GA_ReadOnly)
    gdal.Translate(f"{save_loc}/input.tif", inDs, srcWin=window)
    inDs = None

if __name__ == '__main__':
    reference_raster_path = '/app/VALID/tmp/2024/22/3/-M6_20240122T035117600000Z.nc'
    
    # Generate output label
    label_generator(reference_raster_path)
    crop(reference_raster_path)


