import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, ogr, osr

from fire import Fire

gdal.UseExceptions()


def parse_filename(filename: str) -> dict:
    if filename.startswith("OR_"):
        filename = filename[3:]

    parts = filename.split("_")
    if len(parts) != 5:
        raise ValueError(f"Invalid filename format")

    channel = None

    band_data = parts[0].split("-")
    product = band_data[2]
    if product == "RadC":
        channel = int(parts[0][-2:])

    start_time = parts[2][1:]
    end_time = parts[3][1:]

    start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")
    end_dt = datetime.strptime(end_time, "%Y%j%H%M%S%f")

    return {
        "channel": channel,
        "product": product,
        "start_time": start_dt,
        "end_time": end_dt,
    }


def __process_output(fires:List[Fire], day_path:str) -> None:
    date_split = day_path.split("/")
    date = datetime(year=int(date_split[-3]), month=1, day=1) + timedelta(days=int(date_split[-2]))
    filtered_fires = list(filter(lambda fire: fire.start_date <= date <= fire.end_date, fires))

    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    for fire in filtered_fires:
        print(fire.id, fire.area_acre)
        if fire.geometry.GetGeometryName() == "MULTIPOLYGON":
            for i in range(fire.geometry.GetGeometryCount()):
                multipolygon.AddGeometry(fire.geometry.GetGeometryRef(i))
        elif fire.geometry.GetGeometryName() == "POLYGON":
            multipolygon.AddGeometry(fire.geometry)

    raster_band_500m = None
    for hour in os.listdir(day_path):
        for file in os.listdir(os.path.join(day_path, hour)):
            f = parse_filename(os.path.join(day_path, hour, file).split("/")[-1])
            if f["channel"] == 2:
                raster_band_500m = os.path.join(day_path, hour, file)

    if raster_band_500m == None:
        raise ValueError("500m Band(2) Not found")

    raster_layer = gdal.Open(raster_band_500m)

    cols = raster_layer.RasterXSize
    rows = raster_layer.RasterYSize
    projection = raster_layer.GetProjection()
    geotransform = raster_layer.GetGeoTransform()

    target_layer = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Byte)
    target_layer.SetProjection(projection)
    target_layer.SetGeoTransform(geotransform)

    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("mem_data_source")
    InSR = osr.SpatialReference()
    InSR.SetFromUserInput("ESRI:102498")
    mem_layer = mem_ds.CreateLayer(
        "multipolygon", geom_type=ogr.wkbMultiPolygon, srs=InSR
    )
    feature_defn = mem_layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)

    feature.SetGeometry(multipolygon)
    mem_layer.CreateFeature(feature)

    gdal.RasterizeLayer(
        target_layer, [1], mem_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"]
    )

    gdal.Translate(os.path.join(day_path, "output.tiff"), target_layer, format="GTiff")


def __convert_to_tif(file_path: str):
    layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, "Rad"))
    options = gdal.TranslateOptions(format="GTiff",
                                    width=10000,
                                    height=6000)
    file_name = file_path.replace('.nc', '.tif')
    gdal.Translate(file_name, layer, options=options)
    os.remove(file_path)

    return file_name

def process_day(day_path: str, fires:List[Fire]):
    __process_output(fires, day_path)
    # for hour in os.listdir(day_path):
    #     for file in os.listdir(os.path.join(day_path, hour)):
            # file_path = os.path.join(day_path, hour, file)
            # file_path = __convert_to_tif(file_path)
            # __process_band_file(file_path)


def __process_band_file(file_path: str):
    file = parse_filename(file_path.split("/")[-1])

    if file["product"] != "RadC":
        raise NotImplementedError

    if file["channel"] <= 6:
        __process_Reflectance(file_path)

    else:
        __process_brightness_temperatures(file_path)

    return


def __process_Reflectance(file_path: str, band="Rad"):
    raster_layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, band))
    ds = Dataset(file_path)

    kappa = ds.variables["kappa0"][:]
    Field = kappa * ds.variables["Rad"][:]

    os.remove(file_path)
    driver = gdal.GetDriverByName("netCDF")
    output_dataset = driver.Create(
        file_path,
        raster_layer.RasterXSize,
        raster_layer.RasterYSize,
        1,
        gdal.GDT_Float32,
    )

    # Copy geotransform and projection
    output_dataset.SetGeoTransform(raster_layer.GetGeoTransform())
    output_dataset.SetProjection(raster_layer.GetProjection())

    # Write the calculated data to the new GeoTIFF file
    output_dataset.GetRasterBand(1).WriteArray(Field)
    output_dataset.FlushCache()

    raster_layer = None
    output_dataset = None
    Dataset.close(ds)

    return


def __process_brightness_temperatures(file_path: str, band="Rad"):
    raster_layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, band))
    ds = Dataset(file_path)

    planck_fk1 = ds.variables["planck_fk1"][:]
    planck_fk2 = ds.variables["planck_fk2"][:]
    planck_bc1 = ds.variables["planck_bc1"][:]
    planck_bc2 = ds.variables["planck_bc2"][:]
    Field = (
        planck_fk2 / (np.log((planck_fk1 / ds.variables["Rad"][:]) + 1)) - planck_bc1
    ) / planck_bc2

    os.remove(file_path)

    driver = gdal.GetDriverByName("netCDF")
    output_dataset = driver.Create(
        file_path,
        raster_layer.RasterXSize,
        raster_layer.RasterYSize,
        1,
        gdal.GDT_Float32,
    )

    # Copy geotransform and projection
    output_dataset.SetGeoTransform(raster_layer.GetGeoTransform())
    output_dataset.SetProjection(raster_layer.GetProjection())

    # Write the calculated data to the new GeoTIFF file
    output_dataset.GetRasterBand(1).WriteArray(Field)
    output_dataset.FlushCache()

    raster_layer = None
    output_dataset = None
    Dataset.close(ds)

    return
