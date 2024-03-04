from datetime import datetime, timedelta
import shutil
import os
import time
import logging
from typing import List
from tqdm import tqdm

from osgeo import osr
import s3fs
import botocore

from bbox import Point, Bbox, Bboxs

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="goes_downloader.log", 
    filemode="w"
)

class Downloader:
    def __init__(self, save_dir, bands:List[int], read_bbox_datetime:bool=False) -> None:
        self.fs = s3fs.S3FileSystem(anon=True)
        self.root_dir = f"{save_dir}"
        self.boxes = Bboxs.read_file(read_bbox_datetime).boxes
        self.hour_freq = 1
        self.max_retries = 5
        self.tmp_dir = "tmp"
        self.json_file = "cloud.json"
        self.layers = bands

        if read_bbox_datetime:
            self.hour_freq = None # Since we are downloading all available images in an hour

        self.__convert_to_WGS__()

        if not os.path.exists(self.root_dir):
            os.mkdir(f"{self.root_dir}")

        for box in self.boxes:
            if not os.path.exists(f"{self.root_dir}/{box.id}"):
                os.mkdir(f"{self.root_dir}/{box.id}")

    def clean_root_dir(self, param):
        tmp_dir_path = os.path.join(self.root_dir, self.tmp_dir)

        if param == 'ABI-L2-ACMC':
            tmp_dir_path = os.path.join(self.root_dir, 'cloud_mask')

        if os.path.exists(tmp_dir_path):
            shutil.rmtree(tmp_dir_path)
        logging.info(f"Removed tmp directory- {tmp_dir_path}")

    def point_coversion(self, coord: Point):
        InSR = osr.SpatialReference()
        InSR.SetFromUserInput("EPSG:4326")
        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        transform_epsg = osr.CoordinateTransformation(InSR, OutSR)
        transformed = transform_epsg.TransformPoint(coord.y, coord.x)
        return Point(transformed[0], transformed[1])


    def __convert_to_WGS__(self):
        boxes = []
        for box in self.boxes:
            box_arr = []
            for point in box.box:
                # p = self.point_coversion(point)
                box_arr.append(point)
            boxes.append(Bbox(box_arr, box.id, box.path, box.start, box.end))
        self.boxes = Bboxs(boxes).boxes


    def parse_filename(self, filename: str) -> dict:
        if filename.startswith("OR_"):
            filename = filename[3:] 

        parts = filename.split('_')
        if len(parts) != 5:
            raise ValueError(f"Invalid filename format")

        channel = parts[0][-3:]
        satellite_id = parts[1]
        start_time = parts[2][1:]
        end_time = parts[3][1:]
        creation_time = parts[4][1:-3]

        start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")
        end_dt = datetime.strptime(end_time, "%Y%j%H%M%S%f")
        creation_dt = datetime.strptime(creation_time, "%Y%j%H%M%S%f")

        parameters = {
            "channel": channel,
            "satellite_id": satellite_id,
            "start_time": start_dt,
            "end_time": end_dt,
            "creation_time": creation_dt,
        }
        return parameters

    def filename(self, file):
        par = self.parse_filename(file.replace(".tif", ""))
        file_path = f"{par['channel']}_{par['start_time'].year}{str(par['start_time'].month).zfill(2)}{str(par['start_time'].day).zfill(2)}T{str(par['start_time'].hour).zfill(2)}{str(par['start_time'].minute).zfill(2)}{str(par['start_time'].second).zfill(2)}{str(par['start_time'].microsecond).zfill(3)}Z.tif"
        return file_path

    def download(self, start:datetime, end:datetime, param:str):

        logging.info(f"Starting download for date interval: {start} - {end}")

        dates = [start + timedelta(days=x) for x in range((end - start).days)]
        dates.append(end)
        for date in (pbar:= tqdm(dates, position=0)):
            pbar.set_postfix_str(f"date: {str(date)}")
            day = (datetime(date.year, date.month, date.day) - datetime(date.year, 1, 1)).days + 1

            base_download_dir = os.path.join(self.root_dir, self.tmp_dir)
            if not os.path.exists(base_download_dir):
                os.mkdir(base_download_dir)

            base_download_dir = os.path.join(base_download_dir, str(date.year))
            if not os.path.exists(base_download_dir):
                os.mkdir(base_download_dir)

            try:
                database_hour = self.fs.ls(f"s3://noaa-goes16/{param}/{date.year}/{str(day).zfill(3)}")
            except Exception as e:
                logging.error(f"Unable to query aws for {str(day).zfill(3)}: {param}")
                raise ValueError(f"Unable to load aws due to {e}")

            # Not downloading for hours that lie before start date hour and that lie after end date hour.
            if date == start:
                database_hour = list(filter(lambda e: int(e.split("/")[-1]) >= start.hour, database_hour))
            elif date == end:
                database_hour = list(filter(lambda e: int(e.split("/")[-1]) <= end.hour, database_hour))

            hour = list(map(lambda x: int(x.split("/")[-1]), database_hour))

            day_download_dir = os.path.join(base_download_dir, str(day))
            os.mkdir(day_download_dir)

            for hr in (h_bar := tqdm(hour, leave=False, position=1)):
                h_bar.set_postfix_str(f"hr:{str(hr)}")
                files = self.fs.ls(f"s3://noaa-goes16/{param}/{date.year}/{str(day).zfill(3)}/{str(hr).zfill(2)}/")
                hour_download_dir = os.path.join(day_download_dir, str(hr))
                if self.layers is not None:
                    files = list(filter(lambda x: int(self.parse_filename(x.split("/")[-1])["channel"][1:]) in self.layers, files))
                logging.info(f"Downloading files for {day}:{hr}")

                if not os.path.exists(hour_download_dir):
                    os.mkdir(hour_download_dir)
                else:
                    # checking if all files have been downloaded in last attempt
                    if len(os.listdir(hour_download_dir)) == len(files):
                        logging.info(f"files already present in {hour_download_dir}, skipping download for this.")
                        continue

                retries = 0
                while retries < self.max_retries:

                    try:
                        logging.debug(f"Downloading files- {files}\n{hour_download_dir}")
                        self.fs.get(files, f"{hour_download_dir}/")
                        logging.info(f"Files have been downloaded")
                        break
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == 'Throttling':
                            backoff_time = (2 ** retries) * 120 # 120 seconds as base time
                            logging.warning(f"Throttling detected. Retrying in {backoff_time} seconds.")
                            time.sleep(backoff_time)
                            retries += 1
                        else:
                            # For other errors, raise the exception
                            logging.error(f"Unable to Download aws data for {day}: {param}")
                            raise e
                else:
                    raise Exception(f"Failed to download file after {self.max_retries} retries.")

                for file in os.listdir(hour_download_dir):
                    source = os.path.join(hour_download_dir, file)
                    dst = os.path.join(hour_download_dir, self.filename(file).replace('.tif', '.nc'))
                    os.rename(source, dst)
