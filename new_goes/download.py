import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import botocore
import numpy as np
import s3fs
import skimage
from osgeo import gdal, ogr, osr
from tqdm import tqdm

import preprocess
from fire import Fire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="goes_downloader.log",
    filemode="w",
)


class Downloader:
    """
    Downloads data for fire events.

    Attributes
    ----------
    fires : List[Fire]
        List of Fire objects representing fire events.
    save_dir : str
        Directory path to save downloaded data.
    params : List[str]
        List of parameters for data download.
    """

    def __init__(self, fires: List[Fire], save_dir: str, params: List[str]) -> None:
        """
        Initialize Downloader object.

        Parameters
        ----------
        fires : List[Fire]
            List of Fire objects representing fire events.
        save_dir : str
            Directory path to save downloaded data.
        params : List[str]
            List of parameters for data download.
        """

        self.fires = fires
        self.fs = s3fs.S3FileSystem(anon=True)
        self.root_dir = f"{save_dir}"
        self.params = params
        self.max_retries = 5
        self.patch_file = os.path.join(self.root_dir, "patches")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(self.patch_file):
            os.mkdir(self.patch_file)


    def __datetime_dictionary_list(self, day_bands:List[str]) -> Dict[datetime, List[str]]:
        dict = defaultdict(list)
        for band in day_bands:
            f = preprocess.parse_filename(band.split("/")[-1])
            dict[f["start_time"]].append(band)

        return dict
        
    def __datetime_dictionary(self, day_path: str) -> Dict[datetime, List[str]]:
        dict = defaultdict(list)
        for hour in os.listdir(day_path):
            if not os.path.isdir(os.path.join(day_path, hour)):
                continue
            for file in os.listdir(os.path.join(day_path, hour)):
                f = preprocess.parse_filename(file)
                dict[f["start_time"]].append(os.path.join(day_path, hour, file))
        return dict

    def process_contours(self, contours, day_path:str, fires:List[Fire], win_size=32):
        datetime_dict = self.__datetime_dictionary(day_path)
        for date, files in datetime_dict.items():
            for i, contour in enumerate(contours):
                if len(fires) == 1:
                    fire_id_path = os.path.join(self.patch_file, str(fires[0].id))
                    if not os.path.exists(fire_id_path):
                        os.mkdir(fire_id_path)
                    patch_dir = os.path.join(fire_id_path, str(date))

                else:
                    patch_dir = os.path.join(self.patch_file, str(date))

                os.mkdir(patch_dir)

                centre_x, centre_y = np.mean(contour, axis=0).astype(np.uint64)

                if len(fires) == 1:
                    x_offset = centre_y - win_size // 2
                    y_offset = centre_x - win_size // 2 

                else:
                    x_random = int(random.uniform(-1 * (win_size // 2), win_size // 2))
                    y_random = int(random.uniform(-1 * (win_size // 2), win_size // 2))
                    x_offset = centre_y - win_size // 2 + x_random
                    y_offset = centre_x - win_size // 2 + y_random

                window = (x_offset, y_offset, win_size, win_size)

                for file in files:
                    file_path = os.path.join(patch_dir, f"{file.split('/')[-1]}")
                    gdal.Translate(file_path, file, srcWin=window)
                    # preprocess.process_band_file(file_path)

                output_tiff = os.path.join(day_path, "output.tiff")
                gdal.Translate(os.path.join(patch_dir, "output.tiff"),output_tiff , srcWin=window)

    

    def __process_file(self, file_path: str):
        file_path = preprocess.process_band_file(file_path)
        file_path = preprocess.convert_to_tiff(file_path)


    def process_day(self, day_path: str, curr_fire:Fire|None=None):

        if curr_fire is None:
            fires = self.fires
        else:
            fires = [curr_fire]

        def process_hour(hour):
            hour_path = os.path.join(day_path, hour)
            files = os.listdir(hour_path)
            file_paths = [os.path.join(hour_path, file) for file in files]

            thread_map(self.__process_file, file_paths)

        for hour in os.listdir(day_path):
            process_hour(hour)

        output_location = preprocess.process_output(fires, day_path)
        ds = gdal.Open(output_location)
        myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
        contours = skimage.measure.find_contours(myarray)
        self.process_contours(contours, day_path, fires)

    def __day_cleanup(self, day_path:str):
        os.remove(os.path.join(day_path, "output.tiff"))
        for hour in os.listdir(day_path):
            for file in tqdm(os.listdir(os.path.join(day_path, hour))):
                os.remove(os.path.join(day_path, hour, file))
            os.rmdir(os.path.join(day_path, hour))
        os.rmdir(os.path.join(day_path))

    def download(self, raster_curr_fire:bool=False):
        """
        Downloads data for fire events.
        """

        for fire in (tbar := tqdm(self.fires, position=0)):
            logging.info(f"Starting download for fire: {fire.id}")
            tbar.set_description_str(f"fire: {fire.id}")
            dates = [
                fire.start_date + timedelta(days=x)
                for x in range((fire.end_date - fire.start_date).days + 1)
            ]
            if len(dates) > 2:
                dates = dates[:2]
            for date in (pbar := tqdm(dates, position=1, leave=False)):
                pbar.set_postfix_str(f"date: {str(date)}")
                # if date.month == 7:
                    # WARN:Only August most change for dataset generation
                day_path = self._download_day(date)
                # day_path = temp_download()
                if day_path is not None:
                    if raster_curr_fire:
                        self.process_day(day_path, fire)
                    else:
                        self.process_day(day_path)
                    self.__day_cleanup(day_path)

    def _download_day(self, day: datetime) -> str | None:
        """
        Download data for a specific day.

        Parameters
        ----------
        day : datetime
            Date for which to download data.

        Returns
        -------
        str or None
            Path to the downloaded data directory if successful, otherwise None.
        """
        logging.info(f"Starting download for date: {day}")

        year_path = os.path.join(self.root_dir, str(day.year))
        if not os.path.exists(year_path):
            os.mkdir(year_path)

        days_since_year_start = (
            datetime(day.year, day.month, day.day) - datetime(day.year, 1, 1)
        ).days + 1
        day_path = os.path.join(year_path, str(days_since_year_start))
        if os.path.exists(day_path):
            return None
        os.mkdir(day_path)

        for param in self.params:
            try:
                database_hour = self.fs.ls(
                    f"s3://noaa-goes16/{param}/{day.year}/{str(days_since_year_start).zfill(3)}"
                )
            except Exception as e:
                logging.error(f"Unable to query aws for {str(day).zfill(3)}: {param}")
                raise ValueError(f"Unable to load aws due to {e}")

            hour = list(map(lambda x: int(x.split("/")[-1]), database_hour))

            for hr in (h_bar := tqdm(hour, leave=False, position=2)):
                h_bar.set_postfix_str(f"hr:{str(hr)}")
                h_bar.set_description_str(f"param:{param}")
                files = self.fs.ls(
                    f"s3://noaa-goes16/{param}/{day.year}/{str(days_since_year_start).zfill(3)}/{str(hr).zfill(2)}/"
                )
                hour_download_dir = os.path.join(day_path, str(hr))
                if not os.path.exists(hour_download_dir):
                    os.mkdir(hour_download_dir)

                retries = 0
                while retries < self.max_retries:
                    try:
                        logging.debug(
                            f"Downloading files- {files}\n{hour_download_dir}"
                        )
                        files_dict = self.__datetime_dictionary_list(files)
                        file_timestamp = list(files_dict.keys())[0]
                        self.fs.get(files_dict[file_timestamp], f"{hour_download_dir}/")
                        logging.info("Files have been downloaded")
                        break
                    except botocore.exceptions.ClientError as e:
                        if e.response["Error"]["Code"] == "Throttling":
                            backoff_time = (
                                2**retries
                            ) * 120  # 120 seconds as base time
                            logging.warning(
                                f"Throttling detected. Retrying in {backoff_time} seconds."
                            )
                            time.sleep(backoff_time)
                            retries += 1
                        else:
                            # For other errors, raise the exception
                            logging.error(
                                f"Unable to Download aws data for {day}: {param}"
                            )
                            raise e
                else:
                    raise Exception(
                        f"Failed to download file after {self.max_retries} retries."
                    )
        return day_path


def read_json_file(path: str) -> List[Fire]:
    """
    Read fire data from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing fire data.

    Returns
    -------
    List[Fire]
        A list of Fire objects parsed from the JSON file.
    """

    fires: List[Fire] = []
    with open(path) as f:
        data = json.load(f)
        for fire_data in data["features"]:
            area = fire_data["properties"]["poly_GISAcres"]
            start = fire_data["properties"]["poly_PolygonDateTime"]
            end = fire_data["properties"]["attr_FireOutDateTime"]
            if area < 10.0 or start is None or end is None:
                continue
            fire = Fire.parse(fire_data)
            fires.append(fire)
    return fires


if __name__ == "__main__":
    list_of_strings = lambda arg: list(arg.split(","))
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", required=True)
    parser.add_argument("-j", "--json", required=True)
    parser.add_argument("-p", "--params", required=False, type=list_of_strings)

    args = parser.parse_args()

    fires = read_json_file(args.json)

    down = Downloader(fires, args.save, args.params)
    down.download()
