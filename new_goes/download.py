import argparse
import copy
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List

import botocore
import s3fs
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


def temp_download():
    return "./temp/2023/185/"


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
        # if not os.path.exists(self.patch_file):
        #     os.mkdir(self.patch_file)

    def temp(self):
        day_path = temp_download()
        preprocess.process_day(day_path, self.fires)

    def download(self):
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
                if date.month == 7:
                    # WARN:Only August most change for dataset generation
                    # day_path = self._download_day(date)
                    day_path = temp_download()
                    if day_path is not None:
                        preprocess.process_day(day_path, self.fires)

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
                        self.fs.get(files, f"{hour_download_dir}/")
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
            start = fire_data["properties"]["attr_FireDiscoveryDateTime"]
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
    down.temp()
