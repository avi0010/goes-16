from datetime import datetime
import shutil
import os
import time
import logging

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
    def __init__(self, save_dir, read_bbox_datetime:bool=False) -> None:
        self.fs = s3fs.S3FileSystem(anon=True)
        self.root_dir = f"{save_dir}"
        self.boxes = Bboxs.read_file(read_bbox_datetime).boxes
        self.hour_freq = 1
        self.max_retries = 5
        self.tmp_dir = "tmp"
        self.json_file = "cloud.json"
        self.layers = [7, 12, 13, 14, 15]

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

    def download(self, start:datetime, end:datetime, param:str, latest:bool=False):

        logging.info(f"Starting download for date interval: {start} - {end}")

        # Check Year
        try:
            database_year = self.fs.ls(f"s3://noaa-goes16/{param}/")
            file_param_year = [int(x.split("/")[-1]) for x in database_year] 
        except Exception as e:
            raise ValueError(f"Unable to load aws due to {e}")
        if start.year != end.year:
            raise ValueError(f"{start.year} and {end.year} should be same")
        if not start.year in file_param_year:
            raise ValueError(f"{start.year} not in database")
        if not end.year in file_param_year:
            raise ValueError(f"{end.year} not in database")


        # Check day
        start_date_in_year = (datetime(start.year, start.month, start.day) - datetime(start.year, 1, 1)).days + 1
        end_date_in_year = (datetime(end.year, end.month, end.day) - datetime(start.year, 1, 1)).days + 1

        try:
            database_day = self.fs.ls(f"s3://noaa-goes16/{param}/{start.year}")
            file_param_day = [int(x.split("/")[-1]) for x in database_day]
        except Exception as e:
            raise ValueError(f"Unable to load aws due to {e}")
        if not end_date_in_year in file_param_day:
            file_param_day.pop(-1)
            start_date_in_year -= 1
            end_date_in_year -= 1

        base_download_dir = os.path.join(self.root_dir, self.tmp_dir)
        if param == 'ABI-L2-ACMC':
            base_download_dir = os.path.join(self.root_dir, 'cloud_mask')
        if not os.path.exists(base_download_dir):
            os.mkdir(base_download_dir)


        for day in range(start_date_in_year, end_date_in_year + 1):
            try:
                day_str = '0' * (3 - len(str(day))) + str(day)
                database_hour = self.fs.ls(f"s3://noaa-goes16/{param}/{start.year}/{day_str}")
                file_param_hour = [int(x.split("/")[-1]) for x in database_hour]
            except Exception as e:
                logging.error(f"Unable to query aws for {day_str}: {param}")
                raise ValueError(f"Unable to load aws due to {e}")

            if latest:
                hour = [file_param_hour[-1]]
            if self.hour_freq is None:
                hour = file_param_hour # Because we want to download all images available within an hour
            else:
                hour = file_param_hour[::self.hour_freq]

            day_download_dir = os.path.join(base_download_dir, str(day))
            if not os.path.exists(day_download_dir):
                os.mkdir(day_download_dir)
                logging.info(f"{day_download_dir} created")

            # Not downloading for hours that lie before start date hour and that lie after end date hour.
            if start_date_in_year == end_date_in_year:
                hour = list(filter(lambda e: e > start.hour and e < end.hour, hour))
            if day == start_date_in_year:
                start_hour = start.hour
                hour = list(filter(lambda e: e > start.hour, hour))
            elif day == end_date_in_year:
                end_hour = end.hour
                hour = list(filter(lambda e: e < end.hour, hour))

            for hr in hour:
                hour_download_dir = os.path.join(day_download_dir, str(hr))

                files = self.fs.ls(f"s3://noaa-goes16/{param}/{start.year}/{day_str}/{str(hr).zfill(2)}/")
                files = list(filter(lambda x: int(self.parse_filename(x.split("/")[-1])["channel"][1:]) in self.layers, files))
                logging.info(f"Downloading files for {day}:{hr}")

                if not os.path.exists(hour_download_dir):
                    os.mkdir(hour_download_dir)
                else:
                    # Checking if all files have been downloaded in last attempt
                    if len(os.listdir(hour_download_dir)) == len(files):
                        logging.info(f"Files already present in {hour_download_dir}, skipping download for this.")
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
                    raise Exception(f"Failed to download file after {max_retries} retries.")
                
                for file in os.listdir(hour_download_dir):
                    source = os.path.join(hour_download_dir, file)
                    dst = os.path.join(hour_download_dir, self.filename(file).replace('.tif', '.nc'))
                    os.rename(source, dst)
