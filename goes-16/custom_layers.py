from osgeo import gdal
import numpy as np
from osgeo.gdalconst import GDT_Float32

def wildfire_area(in_file, save_loc, default_confidence_value=0):
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

    inDs = gdal.Open(f"NETCDF:{in_file}:{'Mask'}", gdal.GA_ReadOnly)
    band1 = inDs.GetRasterBand(1)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    cropData = band1.ReadAsArray(0,0,cols,rows)

    outData = np.ones(cropData.shape, dtype=np.float32) * default_confidence_value
    for v, conf in conf_map.items():
        outData[cropData == v] = conf

    fire_indexes = np.where(outData == 1)
    print(fire_indexes)

    inDs = None
    '''outBand.WriteArray(outData)
    outBand.FlushCache()
    outBand.SetNoDataValue(default_confidence_value)
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    del outData'''
