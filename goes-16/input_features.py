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
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="input_features.log", 
    filemode="w"
)

class TiffFile:
    def __init__(self) -> None:
        self.band: int
        self.date: datetime
        self.file_name:str

    def __str__(self) -> str:
        return f"{self.band, self.date, self.file_name}"

    @classmethod
    def parse(cls, name:str):
        cls.file_name = name
        split = name.split("_")
        cls.band = int(split[0][1:])
        year = split[1][0:4]
        month = split[1][4:6]
        day = split[1][6:8]
        hour = split[1][9:11]
        minute = split[1][11:13]
        seconds = split[1][13:15]
        microseconds = split[1][15:split[1].find('Z')]
        date_time = datetime(year=int(year),
                             month=int(month),
                             day=int(day),
                             hour=int(hour),
                             minute=int(minute),
                             second=int(seconds),
                             microsecond=int(microseconds))
        cls.date = date_time
        return cls

class PastFeatures:
    def __init__(self, save_dir, box, date:datetime, past:int, srcwin:int=32) -> None:
        self.fs = s3fs.S3FileSystem(anon=True)
        self.past = past
        self.date = date
        self.box = box
        self.boxes = Bboxs.read_file(False).boxes
        self.save_dir = save_dir
        self.tmp_dir = "./tmp"
        self.tmp_save_dir = "tmp"
        self.srcwin = srcwin
        os.mkdir(self.tmp_dir)
        self.save_dir_cropped = f"{self.tmp_dir}/crop"
        os.mkdir(self.save_dir_cropped)
        for box in self.boxes:
            if int(box.id) == int(self.box.id):
                self.box = box

        self.dates = [date - timedelta(days=x) for x in range(self.past)]


    def search(self):
        files = []

        for date in self.dates:
            day = (datetime(date.year, date.month, date.day) - datetime(date.year, 1, 1)).days + 1
            #convert days in list int to
            days = list(map(lambda x: int(x), list(os.listdir(os.path.join(self.save_dir, self.tmp_dir, str(date.year))))))
            if day not in days:
                continue
            for file in os.listdir(os.path.join(self.save_dir, self.tmp_save_dir, str(date.year), str(day), str(date.hour))):
                directory = os.path.join(self.save_dir, self.tmp_save_dir, str(date.year), str(day), str(date.hour))
                f = TiffFile.parse(file)
                if f.date.minute == self.date.minute and f.date.hour == self.date.hour and f.band == 7:
                    files.append(os.path.join(directory, file))

        return files

    def process(self, window):
        files = self.search()
        arr = []
        for file in files:
            dst = os.path.join(self.save_dir_cropped, file.split("/")[-1])
            gdal.Translate(dst, file, srcWin=window)
            img = gdal.Open(dst)
            arr.append(img.GetRasterBand(1).ReadAsArray())
            os.remove(dst)

        res = np.array(arr).mean(axis=0)
        im = Image.fromarray(res)
        dst = os.path.join(self.save_dir, str(self.box.id), str(self.date), "band_5.tiff")
        im.save(dst)
        os.rmdir(self.save_dir_cropped)
        os.rmdir(self.tmp_dir)


class InputFeatures:
    def __init__(self, save_dir, past:int, win_size) -> None:
        self.layers=[[7, 12],
                     [7, 13],
                     [7, 14],
                     [7, 15]]

        self.input_layers = [7, 12, 13, 14, 15]
        self.boxes = Bboxs.read_file(True).boxes
        self.save_dir = save_dir
        self.tmp_dir = "tmp"
        self.datetime_images = defaultdict(list)
        self.past = past
        self.win_size = win_size
        
        for year in os.listdir(f"{self.save_dir}/{self.tmp_dir}"):
            for day in os.listdir(f"{self.save_dir}/{self.tmp_dir}/{year}"):
                for hour in os.listdir(f"{self.save_dir}/{self.tmp_dir}/{year}/{day}"):
                   directory = os.path.join(self.save_dir, self.tmp_dir, year, str(day), str(hour))
                   for file in os.listdir(directory):
                       tif_file = TiffFile.parse(file)
                       if tif_file.band in self.input_layers:
                           self.datetime_images[tif_file.date].append(file)

        for box in self.boxes:
            if not os.path.exists(f"{self.save_dir}/{box.id}"):
                os.mkdir(f"{self.save_dir}/{box.id}")

            for date in self.datetime_images.keys():
                if box.start < date and date < box.end:
                    os.mkdir(f"{self.save_dir}/{box.id}/{str(date)}")

    def create_polygon(self, geojson_data, box:Bbox):
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for poly in geojson_data["features"]:
            if poly["properties"]["poly_SourceOID"] == int(box.id):
                geom = ogr.CreateGeometryFromJson(json.dumps(poly['geometry']))
                if geom.GetGeometryName() == 'MULTIPOLYGON':
                    for i in range(geom.GetGeometryCount()):
                        multipolygon.AddGeometry(geom.GetGeometryRef(i))
                elif geom.GetGeometryName() == 'POLYGON':
                    multipolygon.AddGeometry(geom)

        return multipolygon


    def get_center_pixel(self, label_path, mask_value:int = 1):
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

    def model_inputs(self):
        for box in self.boxes:
            for time_stamp in os.listdir(os.path.join(self.save_dir, str(box.id))):
                if time_stamp.find('.') == -1:
                    parsed_date = datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                else:
                    parsed_date = datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S.%f")

                for idx, layer in enumerate(self.layers):
                    layer1, layer2 = None, None
                    for file in self.datetime_images[parsed_date]:
                        tf = TiffFile.parse(file)
                        directory = os.path.join(self.save_dir, str(box.id), time_stamp)
                        if len(layer) == 1 and tf.band == 7:
                            img = gdal.Open(os.path.join(directory, file))
                            layer1 = np.array(img.GetRasterBand(1).ReadAsArray())
                            im = Image.fromarray(layer1)
                            save_location = os.path.join(self.save_dir, str(box.id), time_stamp, f"input_1.tif")
                            continue

                        if tf.band == layer[0]:
                            img = gdal.Open(os.path.join(directory, file))
                            layer1 = np.array(img.GetRasterBand(1).ReadAsArray())

                        if tf.band == layer[1]:
                            img = gdal.Open(os.path.join(directory, file))
                            layer2 = np.array(img.GetRasterBand(1).ReadAsArray())

                    
                    band = layer1 - layer2
                    im = Image.fromarray(band)
                    save_location = os.path.join(self.save_dir, str(box.id), time_stamp, f"input_{idx + 1}.tif")
                    im.save(save_location)


    def input_features(self):
        for box in (pbar := tqdm(self.boxes, desc="Input features")):
            pbar.set_description_str(f"Box: {str(box.id)}")
            for time_stamp in tqdm(os.listdir(os.path.join(self.save_dir, str(box.id))), leave=False, desc="time"):
                if time_stamp.find('.') == -1:
                    parsed_date = datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                else:
                    parsed_date = datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S.%f")

                mask_file = os.path.join(self.save_dir, str(box.id), time_stamp, "output.tif")
                centre_x, centre_y = self.get_center_pixel(mask_file)

                # Finding offset by moving the center pixels to top-left according to image size
                x_random = int(random.uniform(-1 * (self.win_size // 3), self.win_size // 3))
                y_random = int(random.uniform(-1 * (self.win_size // 3), self.win_size // 3))
                x_offset = centre_y - self.win_size // 2 + x_random
                y_offset = centre_x - self.win_size // 2 + y_random

                window = (x_offset, y_offset, self.win_size, self.win_size)

                # Translate output.tif
                gdal.Translate(mask_file.replace("tif", "tiff"), mask_file, srcWin=window)
                os.remove(mask_file)
                past = PastFeatures(self.save_dir, box, parsed_date, past=self.past, srcwin=self.win_size)
                past.process(window)
                logging.info(f"Past data processed - Box- {box.id} | Timestamp- {time_stamp}")

                date_from_year_start = (parsed_date - datetime(year=parsed_date.year, month=1, day=1))
                for file in self.datetime_images[parsed_date]:
                    src_file = os.path.join(self.save_dir, self.tmp_dir, str(parsed_date.year), str(date_from_year_start.days + 1), str(parsed_date.hour), file)
                    dst_file = os.path.join(self.save_dir, str(box.id), time_stamp, file)
                    gdal.Translate(dst_file, src_file, srcWin=window)
                    logging.info(f"File cropped - Box- {box.id} | Timestamp- {time_stamp} | Src File- {src_file}")    
                logging.info(f"Input feature generated for box- {box.id} & timestamp- {time_stamp}")
            logging.info(f"Input feature generation completed for box- {box.id}")

    def output_label(self):

        with open("./files/reprojected_NIFC_2023_Wildfire_Perimeters.json") as f:
            geojson_data = json.load(f)

        for box in (pbar := tqdm(self.boxes, leave=False)):
            pbar.set_postfix_str(f"Box: {str(box.id)}")
            for date in tqdm(self.datetime_images.keys(), desc="date", leave=False):
                if box.start < date and date < box.end:

                    multipolygon = self.create_polygon(geojson_data, box)
                    image_name = self.datetime_images[date][0]
                    date_from_year_start = (date - datetime(year=date.year, month=1, day=1)).days + 1
                    reference_raster_path = os.path.join(self.save_dir, self.tmp_dir, str(date.year), str(date_from_year_start), str(date.hour), image_name)

                    raster_layer = gdal.Open(reference_raster_path)
                    cols = raster_layer.RasterXSize
                    rows = raster_layer.RasterYSize
                    projection = raster_layer.GetProjection()
                    geotransform = raster_layer.GetGeoTransform()

                    target_layer = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
                    target_layer.SetProjection(projection)
                    target_layer.SetGeoTransform(geotransform)

                    mem_driver = ogr.GetDriverByName('Memory')
                    mem_ds = mem_driver.CreateDataSource('mem_data_source')
                    InSR = osr.SpatialReference()
                    InSR.SetFromUserInput("ESRI:102498")
                    mem_layer = mem_ds.CreateLayer('multipolygon', geom_type=ogr.wkbMultiPolygon, srs=InSR)
                    feature_defn = mem_layer.GetLayerDefn()
                    feature = ogr.Feature(feature_defn)
                    feature.SetGeometry(multipolygon)
                    mem_layer.CreateFeature(feature)

                    gdal.RasterizeLayer(target_layer, [1], mem_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])

                    output_file = os.path.join(self.save_dir, str(box.id), str(date), "output.tif")
                    gdal.Translate(output_file, target_layer, format='GTiff')

            logging.info(f"Output label generated for box- {box.id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Type of Download")
    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-p", "--past", required=True)
    parser.add_argument("-w", "--window", required=True)

    args = parser.parse_args()
    ip = InputFeatures(args.data, int(args.past), int(args.window))
    try:
        ip.output_label()
        ip.input_features()
    except Exception as e:
        logging.error(str(e))
        raise
