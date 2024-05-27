import os
from multiprocessing import Pool
from typing import List
import s3fs
from osgeo import gdal
from tqdm import tqdm
from netCDF4 import Dataset
from pprint import pprint
import numpy as np
from datetime import datetime
import argparse


PARAM = "ABI-L1b-RadC"


class Downloader:
    def __init__(self) -> None:
        self.fs = s3fs.S3FileSystem(anon=True)
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")

        self.base_download_path = "./tmp/base"
        self.__clean_data()
        if not os.path.exists(self.base_download_path):
            os.mkdir(self.base_download_path)

    def __clean_data(self):
        if os.path.exists(self.base_download_path):
            for file in os.listdir(self.base_download_path):
                os.remove(os.path.join(self.base_download_path, file))

    def parse_filename(self, filename: str) -> dict:
        if filename.startswith("OR_"):
            filename = filename[3:]

        parts = filename.split("_")
        if len(parts) != 5:
            raise ValueError(f"Invalid filename format")

        channel = parts[0][-2:]
        satellite_id = parts[1]
        start_time = parts[2][1:]
        end_time = parts[3][1:]
        creation_time = parts[4][1:-3]

        start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")
        end_dt = datetime.strptime(end_time, "%Y%j%H%M%S%f")
        creation_dt = datetime.strptime(creation_time, "%Y%j%H%M%S%f")

        parameters = {
            "channel": int(channel),
            "satellite_id": satellite_id,
            "start_time": start_dt,
            "end_time": end_dt,
            "creation_time": creation_dt,
        }
        return parameters

    def nearest_min(self, time: datetime, dates: List[datetime]):
        min_diff = float("inf")
        nearest_datetime = None

        for date in dates:
            difference = abs((time - date).total_seconds())

            if difference < min_diff:
                min_diff = difference
                nearest_datetime = date

        if nearest_datetime is None:
            raise ValueError("Unable to find nearest datetime")

        return nearest_datetime

    def process_file(self, file):
        file_path = self.__process_band_file(file)
        self.__convert_to_tiff(file_path)

    def process_files(self):
        with Pool() as pp:
            files = os.listdir(self.base_download_path)
            files_paths = [
                os.path.join(self.base_download_path, file) for file in files
            ]
            with tqdm(total=len(files_paths), leave=False) as pbar:
                for _ in pp.imap_unordered(self.process_file, files_paths):
                    pbar.update()

    def __convert_to_tiff(self, file_path):
        layer = gdal.Open(file_path)
        options = gdal.WarpOptions(
            format="GTiff",
            # srcSRS="ESRI:102498",
            # dstSRS="EPSG:4326",
            width=10000,
            height=6000,
        )
        file_name = file_path.replace(".tiff", ".tif")
        gdal.Warp(file_name, layer, options=options)
        os.remove(file_path)

        return file_name

    def __process_band_file(self, file_path: str, band="Rad") -> str:
        file = self.parse_filename(file_path.split("/")[-1])

        if file["channel"] <= 6:
            raster_layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, band))
            ds = Dataset(file_path)

            kappa = ds.variables["kappa0"][:]
            Field = kappa * ds.variables["Rad"][:]

            os.remove(file_path)
            file_path = file_path.replace(".nc", ".tiff")
            driver = gdal.GetDriverByName("Gtiff")
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

            return file_path

        else:
            raster_layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, band))
            ds = Dataset(file_path)

            planck_fk1 = ds.variables["planck_fk1"][:]
            planck_fk2 = ds.variables["planck_fk2"][:]
            planck_bc1 = ds.variables["planck_bc1"][:]
            planck_bc2 = ds.variables["planck_bc2"][:]
            Field = (
                planck_fk2 / (np.log((planck_fk1 / ds.variables["Rad"][:]) + 1))
                - planck_bc1
            ) / planck_bc2

            os.remove(file_path)
            file_path = file_path.replace(".nc", ".tiff")
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

            return file_path


    def download_latest(self):
        try:

            day_path = self.fs.ls(f"s3://noaa-goes16/{PARAM}/{datetime.now().year}")[-1]
            hour_path = self.fs.ls(day_path)[-1]
            files = self.fs.ls(hour_path)

            files_datetimes = list(map(lambda x: self.parse_filename(x.split("/")[-1])['start_time'], files))
            max_datetime = max(files_datetimes)

            latest_files = []

            for file in files:
                if self.parse_filename(file.split("/")[-1])['start_time'] == max_datetime:
                    latest_files.append(file)

            assert(len(latest_files) == 16)

            self.fs.get(latest_files, self.base_download_path)

            self.process_files()


        except Exception as e:
            raise ValueError(f"Unable to load aws due to {e}")

    def download_datetime(self, day: datetime):
        days_since_year_start = (
            datetime(day.year, day.month, day.day) - datetime(day.year, 1, 1)
        ).days + 1

        try:
            data_hour = self.fs.ls(
                f"s3://noaa-goes16/{PARAM}/{day.year}/{str(days_since_year_start).zfill(3)}/{str(day.hour).zfill(2)}"
            )

            dates = list(
                map(
                    lambda x: self.parse_filename(x.split("/")[-1])["start_time"],
                    data_hour,
                )
            )
            nearest_time = self.nearest_min(day, dates)
            files = list(
                filter(
                    lambda x: f"s{nearest_time.strftime('%Y%j%H%M%S')}" in x, data_hour
                )
            )

            assert len(files) == 16

            self.fs.get(files, self.base_download_path)

            self.process_files()

        except Exception as e:
            print(f"Unable to query aws for {str(day).zfill(3)}: {PARAM}")
            raise ValueError(f"Unable to load aws due to {e}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--date", required=True)
    # args = parser.parse_args()

    # start_dt = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
    down = Downloader()
    down.download_latest()
    # down.download_datetime(start_dt)
