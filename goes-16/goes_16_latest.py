from datetime import datetime
from osgeo import gdal
import numpy as np
import os
import json
from osgeo import osr
from PIL import Image
import logging
import argparse
from Downloader import Downloader
from custom_layers import wildfire_area

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="goes_downloader.log", 
    filemode="a"
)

class GoesDownloaderLatest(Downloader):
    def __init__(self, save_dir) -> None:
        super().__init__(save_dir)

        logging.info("Calculating latest cloud cover")
        self.__bbox_cloud_covers__()

    def wildfire_map(self):
        self.download(datetime.now(), datetime.now(), "ABI-L2-FDCC", latest=True)
        for box in self.boxes:
            if not os.path.exists(f"{self.root_dir}/{box.id}/wld_map"):
                os.mkdir(f"{self.root_dir}/{box.id}/wld_map/")
        day = os.listdir(f"{self.root_dir}/{self.tmp_dir}/")[0]
        hour = os.listdir(f"{self.root_dir}/{self.tmp_dir}//{day}")[0]
        directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hour}/"

        for file in os.listdir(directory):
           wildfire_area(f"{directory}/{file}", directory)
           os.remove(f"{directory}/{file}")

        for file in os.listdir(directory):
            layer = gdal.Open(f"{directory}/{file}")
            options = gdal.TranslateOptions(format="GTiff")
            file_name = file.replace('.nc', '.tif')
            gdal.Translate(f"{directory}/{file_name}", layer, options=options)
            os.remove(f"{directory}/{file}")

        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        for file in os.listdir(directory):
            file_path = self.filename(file)
            for (box, box_file) in self.bbox_cloud_cover.items():
                min_box_file = self.parse_filename(box_file.replace(".tif", ""))["start_time"]
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

    def __bbox_cloud_covers__(self):
        self.download(datetime.now(), datetime.now(), "ABI-L2-ACHAC", latest=True)
        bbox_lowest_cloud_path = {}
        bbox_lowest_cloud_value = {}

        day = os.listdir(f"{self.root_dir}/{self.tmp_dir}/")[0]
        hour = os.listdir(f"{self.root_dir}/{self.tmp_dir}//{day}")[0]
        directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hour}/"
        for file in os.listdir(directory):
            layer = gdal.Open("NETCDF:{0}:{1}".format(f"{directory}/{file}", "DQF"))
            options = gdal.TranslateOptions(format="GTiff")
            file_name = file.replace('.nc', '.tif')
            gdal.Translate(f"{directory}/{file_name}", layer, options=options)
            os.remove(f"{directory}/{file}")

        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")
        for box in self.boxes:
            os.mkdir(f"{self.root_dir}/{self.tmp_dir}/{box.id}")
            cloud_cover, cloud_file = -1, None
            for file in os.listdir(directory):
                options = gdal.WarpOptions(format="GTiff",
                                           srcSRS=OutSR,
                                           dstSRS=OutSR,
                                           cutlineDSName=f"{box.path}",
                                           cropToCutline=True,
                                           copyMetadata=True)

                gdal.Warp(f"{self.root_dir}/{self.tmp_dir}/{box.id}/{file}",
                          f"{directory}/{file}",
                          options=options)

            for bbox in os.listdir(f"{self.root_dir}/{self.tmp_dir}/{box.id}/"):
                im = Image.open(f"{self.root_dir}/{self.tmp_dir}/{box.id}/{bbox}")
                imarray = np.array(im)
                shape = imarray.shape
                unique, counts = np.unique(imarray, return_counts=True)
                nc_dict = dict(zip(unique, counts))
                density = nc_dict.get(0.0, 1) / (shape[0] * shape[1])
                if density > cloud_cover:
                    cloud_cover = max(cloud_cover, density)
                    cloud_file = bbox
            bbox_lowest_cloud_path[box] = cloud_file
            bbox_lowest_cloud_value[box.id] = cloud_cover
        self.bbox_cloud_cover = bbox_lowest_cloud_path
        self.bbox_cloud_value = bbox_lowest_cloud_value
        self.clean_root_dir()

    def run(self, param, save_location, band):
        self.download(datetime.now(), datetime.now(), param, latest=True)
        for box in self.boxes:
            if not os.path.exists(f"{self.root_dir}/{box.id}/{save_location}"):
                os.mkdir(f"{self.root_dir}/{box.id}/{save_location}/")

        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        day = os.listdir(f"{self.root_dir}/{self.tmp_dir}/")[0]
        hour = os.listdir(f"{self.root_dir}/{self.tmp_dir}/{day}")[0]
        directory = f"{self.root_dir}/{self.tmp_dir}/{day}/{hour}/"
        for file in os.listdir(directory):
            layer = gdal.Open("NETCDF:{0}:{1}".format(f"{directory}/{file}", band))
            options = gdal.TranslateOptions(format="GTiff")
            file_name = file.replace('.nc', '.tif')
            gdal.Translate(f"{directory}/{file_name}", layer, options=options)
            os.remove(f"{directory}/{file}")

        for file in os.listdir(directory):
            file_path = self.filename(file)
            for (box, box_file) in self.bbox_cloud_cover.items():
                min_box_file = self.parse_filename(box_file.replace(".tif", ""))["start_time"]
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

    def cloud_json(self):
        if not os.path.exists(f"{self.root_dir}/{self.json_file}"):
            empty_json = {}
            with open(f"{self.root_dir}/{self.json_file}", "w") as f:
                json.dump(empty_json, f)

        with open(f"{self.root_dir}/{self.json_file}", "r") as f:
            data = json.load(f)

        with open(f"{self.root_dir}/{self.json_file}", "w") as f:
            data[str(datetime.now())] = self.bbox_cloud_value
            logging.info("Saving JSON file")
            json.dump(data, f)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Type of Download")
    parser.add_argument("-s", "--save", required=True)

    latest_parser = subparsers.add_parser("latest")

    datetime_parser = subparsers.add_parser("date")
    datetime_parser.add_argument("-d", "--date", nargs=2, type=str, required=True)
    args = parser.parse_args()

    down = GoesDownloaderLatest(args.save)
    down.wildfire_map()
    down.run("ABI-L2-ACHAC", "cloud", "HT")
    down.run("ABI-L2-FDCC", "mask", "Mask")
    #down.run("ABI-L2-FDCC", "area", "Area")
    #down.run("ABI-L2-FDCC", "power", "Power")
    #down.run("ABI-L2-FDCC", "temp", "Temp")
    down.cloud_json()
