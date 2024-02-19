from datetime import datetime
from operator import itemgetter

from osgeo import gdal
import os
from PIL import Image
import numpy as np
import shutil
from custom_layers import wildfire_area
from osgeo import osr
import logging
from Downloader import Downloader

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="goes_downloader.log", 
    filemode="w"
)


class GoesDownloaderDate(Downloader):
    def __init__(self, save_dir, start:datetime=None, end:datetime=None) -> None:
        super().__init__(save_dir)
        self.start = start
        self.end = end

        logging.info("Calculating cloud cover for Bulk Download")
        self.__bbox_cloud_covers__()
        self.__index_bbox__()

    def wildfire_map(self):
        self.download(self.start, self.end, "ABI-L2-FDCC")
        for box in self.boxes:
            if not os.path.exists(f"{self.root_dir}/{box.id}/wld_map"):
                os.mkdir(f"{self.root_dir}/{box.id}/wld_map/")

        for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}/"):
            for hr in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"
                    wildfire_area(f"{directory}/{file}", directory)
                    os.remove(f"{directory}/{file}")

        for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}/"):
            for hr in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"
                    layer = gdal.Open(f"{directory}/{file}")
                    options = gdal.TranslateOptions(format="GTiff")
                    file_name = file.replace('.nc', '.tif')
                    gdal.Translate(f"{directory}/{file_name}", layer, options=options)
                    os.remove(f"{directory}/{file}")

        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}/"):
            for hr in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"
                    file_path = self.filename(file)
                    for (box, box_path) in self.box_file_map.items():
                        min_box_file = self.parse_filename(box_path[day][hr].replace(".tif", ""))["start_time"]
                        if self.parse_filename(file.replace(".tif", ""))["start_time"] != min_box_file:
                            continue

                        options = gdal.WarpOptions(format="GTiff",
                                                   srcSRS=OutSR,
                                                   dstSRS='EPSG:4326',
                                                   cutlineDSName=f"{box.path}",
                                                   cropToCutline=True)

                        gdal.Warp(f"{self.root_dir}/{box.id}/wld_map/{file_path}",
                                  f"{directory}/{file}",
                                  options=options)
        self.clean_root_dir()
                    
    def __index_bbox__(self):
        box_cover_map, box_file_map = {}, {}
        for box in self.boxes:
            day_cover_map, day_file_map = {}, {}
            for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/"):
                hrs_cover_map, hrs_file_map  = {}, {}
                for hr in os.listdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/{day}"):
                    cloud_cover, cloud_file = -1, None
                    for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/{day}/{hr}"):
                        im = Image.open(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/{day}/{hr}/{file}")
                        imarray = np.array(im)
                        shape = imarray.shape
                        unique, counts = np.unique(imarray, return_counts=True)
                        nc_dict = dict(zip(unique, counts))
                        density = nc_dict.get(0.0, 1) / (shape[0] * shape[1])
                        if density > cloud_cover:
                            cloud_cover = max(cloud_cover, density)
                            cloud_file = file
                    hrs_cover_map[hr] = cloud_cover
                    hrs_file_map[hr] = cloud_file
                day_cover_map[day] = hrs_cover_map
                day_file_map[day] = hrs_file_map
            box_cover_map[box] = day_cover_map
            box_file_map[box] = day_file_map
        self.box_cover_map = box_cover_map
        self.box_file_map = box_file_map
        shutil.rmtree(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/")

    def __bbox_cloud_covers__(self):
        self.download(self.start, self.end, "ABI-L2-ACHAC", latest=False)
        for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}"):
            for hour in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}/{hour}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hour}"
                    layer = gdal.Open("NETCDF:{0}:{1}".format(f"{directory}/{file}", "DQF"))
                    options = gdal.TranslateOptions(format="GTiff")
                    file_name = file.replace('.nc', '.tif')
                    gdal.Translate(f"{directory}/{file_name}", layer, options=options)
                    os.remove(f"{directory}/{file}")

        os.mkdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}")
        for box in self.boxes:
            os.mkdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}")
            for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}"):
                os.mkdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/{day}")
                for hour in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                    os.mkdir(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/{day}/{hour}")

        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        for box in self.boxes:
            for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}"):
                for hour in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hour}"
                    for file in os.listdir(directory):
                        options = gdal.WarpOptions(format="GTiff",
                                                   srcSRS=OutSR,
                                                   dstSRS=OutSR,
                                                   cutlineDSName=f"{box.path}",
                                                   cropToCutline=True,
                                                   copyMetadata=True)

                        gdal.Warp(f"{self.root_dir}/{self.tmp_dir}_{self.tmp_dir}/{box.id}/{day}/{hour}/{file}",
                                  f"{directory}/{file}",
                                  options=options)
        self.clean_root_dir()

    def run(self, param, save_location, band):
        self.download(self.start, self.end, param, latest=False)
        for box in self.boxes:
            if not os.path.exists(f"{self.root_dir}/{box.id}/{save_location}"):
                os.mkdir(f"{self.root_dir}/{box.id}/{save_location}/")

        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}/"):
            for hr in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"
                    layer = gdal.Open("NETCDF:{0}:{1}".format(f"{directory}/{file}", band))
                    options = gdal.TranslateOptions(format="GTiff")
                    file_name = file.replace('.nc', '.tif')
                    gdal.Translate(f"{directory}/{file_name}", layer, options=options)
                    os.remove(f"{directory}/{file}")

        for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}/"):
            for hr in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                for file in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"
                    file_path = self.filename(file)
                    for (box, box_path) in self.box_file_map.items():
                        min_box_file = self.parse_filename(box_path[day][hr].replace(".tif", ""))["start_time"]
                        if self.parse_filename(file.replace(".tif", ""))["start_time"] != min_box_file:
                            continue

                        options = gdal.WarpOptions(format="GTiff",
                                                   srcSRS=OutSR,
                                                   dstSRS='EPSG:4326',
                                                   cutlineDSName=f"{box.path}",
                                                   cropToCutline=True)

                        gdal.Warp(f"{self.root_dir}/{box.id}/{save_location}/{file_path}",
                                  f"{directory}/{file}",
                                  options=options)
        self.clean_root_dir()

class GoesDownloaderIndividualBboxDate(Downloader):

    def __init__(self, save_dir) -> None:
        super().__init__(save_dir, True)
        self.start = None
        self.end = None

        self.__date_interval_bboxs__()

    def __date_interval_bboxs__(self):

        date_intervals = [(box.start, box.end) for box in self.boxes]
        self.start = min(date_intervals, key=itemgetter(0))[0]
        self.end = max(date_intervals, key=itemgetter(1))[1]

    def pre_processing(self, param, band):

        logging.info("Converting .nc files to .tif")

        for day in os.listdir(os.path.join(self.root_dir, self.tmp_dir)):
            for hr in os.listdir(os.path.join(self.root_dir, self.tmp_dir, day)):
                for file in os.listdir(os.path.join(self.root_dir, self.tmp_dir, day, hr)):
                    directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hr}"

                    if file.endswith('.nc'):
                        layer = gdal.Open("NETCDF:{0}:{1}".format(f"{directory}/{file}", band))
                        options = gdal.TranslateOptions(format="GTiff")
                        file_name = file.replace('.nc', '.tif')

                        gdal.Translate(f"{directory}/{file_name}", layer, options=options)
                        os.remove(f"{directory}/{file}")
        
        logging.info("Performing cloud masking")
        if param != 'ABI-L2-ACMC':
            # TODO- Use cloud masks (present in args.save/cloud_mask) on these images and perform interpolation to fill no data values
            pass

    def crop_images_for_bboxs_one(self, param, save_location):
       if param == 'ABI-L2-ACMC':
           raise NotImplementedError

       OutSR = osr.SpatialReference()
       OutSR.SetFromUserInput("ESRI:102498")
       base_dir = os.path.join(self.root_dir, self.tmp_dir)

       for box in self.boxes:
           save_location_path = os.path.join(self.root_dir, box.id, save_location)
           if not os.path.exists(save_location_path):
               os.mkdir(save_location_path)

           for day in os.listdir(f"{self.root_dir}/{self.tmp_dir}"):
               for hour in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}"):
                  directory = os.path.join(base_dir, str(day), str(hour))
                  for file in os.listdir(directory):
                    options = gdal.WarpOptions(format="GTiff",
                                           srcSRS=OutSR,
                                           dstSRS='EPSG:4326',
                                           cutlineDSName=f"{box.path}",
                                           copyMetadata= True,
                                           cropToCutline=True)

                    gdal.Warp(os.path.join(save_location_path, file),
                              os.path.join(directory, file),
                              options=options)

    def crop_images_for_bboxs(self, param, save_location):
       if param != 'ABI-L2-ACMC':
            
            OutSR = osr.SpatialReference()
            OutSR.SetFromUserInput("ESRI:102498")
            base_dir = os.path.join(self.root_dir, self.tmp_dir)

            for box in self.boxes:

                save_location_path = os.path.join(self.root_dir, box.id, save_location)
                if not os.path.exists(save_location_path):
                    os.mkdir(save_location_path)

                start_date_in_year = (box.start - datetime(self.start.year, 1, 1)).days + 1
                end_date_in_year = (box.end - datetime(self.start.year, 1, 1)).days + 1

                for day in range(start_date_in_year, end_date_in_year + 1):
                    hours = [i for i in range(0, 24)]
                    if day == start_date_in_year:
                        start_hour = box.start.hour
                        hours = list(filter(lambda e: e > start_hour, hours))
                    elif day == end_date_in_year:
                        end_hour = box.end.hour
                        hours = list(filter(lambda e: e < end_hour, hours))

                    for hour in hours:
                        directory = os.path.join(base_dir, str(day), str(hour))
                        if not os.path.exists(directory):
                            logging.warning(f"Directory- {directory} not found. Check if download script ran properly")
                            continue

                        for file in os.listdir(directory):

                            options = gdal.WarpOptions(format="GTiff",
                                                   srcSRS=OutSR,
                                                   dstSRS='EPSG:4326',
                                                   cutlineDSName=f"{box.path}",
                                                   cropToCutline=True)

                            gdal.Warp(os.path.join(save_location_path, file),
                                      os.path.join(directory, file),
                                      options=options)


if __name__ == "__main__":
    down = GoesDownloaderDate("/tmp/DATA", datetime(2023, 9, 30), datetime(2023, 10, 2))
    down.wildfire_map()
    down.run("ABI-L2-ACHAC", "cloud", "HT")
    down.run("ABI-L2-FDCC", "mask", "Mask")
    down.run("ABI-L2-FDCC", "area", "Area")
    down.run("ABI-L2-FDCC", "power", "Power")
    down.run("ABI-L2-FDCC", "temp", "Temp")
