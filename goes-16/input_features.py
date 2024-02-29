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
        microseconds = split[1][15:21]
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
    def __init__(self, save_dir, box, date:datetime) -> None:
        self.fs = s3fs.S3FileSystem(anon=True)
        self.past = 15
        self.dir_name = "past"
        self.date = date
        self.box = box
        self.boxes = Bboxs.read_file(False).boxes
        self.save_dir = save_dir
        self.tmp_dir = "./tmp"
        os.mkdir(self.tmp_dir)
        self.tmp_dir_past = f"{self.tmp_dir}/{self.dir_name}"
        self.save_dir_cropped = f"{self.tmp_dir}/crop"
        os.mkdir(self.tmp_dir_past)
        os.mkdir(self.save_dir_cropped)
        for box in self.boxes:
            if int(box.id) == int(self.box):
                self.box_path = box.path

    def parse_filename(self, filename: str):
        if filename.startswith("OR_"):
            filename = filename[3:] 

        parts = filename.split('_')
        if len(parts) != 5:
            raise ValueError(f"Invalid filename format")

        start_time = parts[2][1:]
        start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")
        channel = int(parts[0][-2:])

        return start_dt, channel

    def process(self):
        self.download()
        self.convert_to_tif()
        self.crop_file()
        arr = []
        for f in os.listdir(self.save_dir_cropped):
            img = gdal.Open(f"{self.save_dir_cropped}/{f}")
            band = np.array(img.GetRasterBand(1).ReadAsArray())
            band = np.where(band < np.percentile(band, 99), band, 0)
            band = np.where(band > np.percentile(band, 1), band, 0)
            arr.append(band.tolist())
            os.remove(os.path.join(self.save_dir_cropped, f))

        result = np.array(arr).mean(axis=0).astype(np.uint8)
        im = Image.fromarray(result)
        im = im.resize((64, 64))
        im.save(f"{self.save_dir}/{self.box}/{str(self.date)}/b_{5}.tiff")
        os.rmdir(self.save_dir_cropped)
        os.rmdir(self.tmp_dir)

    def crop_file(self) -> None:
        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")
        for file in os.listdir(self.tmp_dir_past):
            options = gdal.WarpOptions(format="GTiff",
                                   srcSRS=OutSR,
                                   dstSRS='EPSG:4326',
                                   cutlineDSName=f"{self.box_path}",
                                   copyMetadata= True,
                                   cropToCutline=True)

            gdal.Warp(os.path.join(self.save_dir_cropped, file),
                      os.path.join(self.tmp_dir_past, file),
                      options=options)

            os.remove(os.path.join(self.tmp_dir_past, file))

        os.rmdir(self.tmp_dir_past)

    def convert_to_tif(self, band="Rad") -> None:
        for file in os.listdir(self.tmp_dir_past):
            if file.endswith('.nc'):
                layer = gdal.Open("NETCDF:{0}:{1}".format(f"{self.tmp_dir_past}/{file}", band))
                options = gdal.TranslateOptions(format="GTiff")
                file_name = file.replace('.nc', '.tif')

                gdal.Translate(f"{self.tmp_dir_past}/{file_name}", layer, options=options)
                os.remove(f"{self.tmp_dir_past}/{file}")

    def download(self) -> None:
        if self.date.month == 1 and self.date.day <= self.past:
            start_date_in_year = datetime(self.date.year, 1, 1).day
        else:
            start_date_in_year = ((datetime(self.date.year, self.date.month, self.date.day) - timedelta(days=self.past)) - datetime(self.date.year, 1, 1)).days + 1

        end_date_in_year = (datetime(self.date.year, self.date.month, self.date.day) - datetime(self.date.year, 1, 1)).days + 1
        past_files = []
        for day in range(start_date_in_year, end_date_in_year):
            day_str = str(day).zfill(3)
            temp = f"s3://noaa-goes16/ABI-L1b-RadC/{self.date.year}/{day_str}/{str(self.date.hour).zfill(2)}"
            hour_files = self.fs.ls(temp)
            for file in hour_files:
                file_data, ch = self.parse_filename(file.split("/")[-1])
                if file_data.minute == self.date.minute and ch == 7:
                    past_files.append(file)

        self.fs.get(past_files, self.tmp_dir_past)

class InputFeatures:
    def __init__(self, save_dir) -> None:
        self.layers=[[7, 12],
                     [7, 13],
                     [7, 14],
                     [7, 15]]

        self.input_layers = [7, 12, 13, 14, 15]
        self.boxes = Bboxs.read_file(True).boxes
        self.save_dir = save_dir
        self.tmp_dir = "tmp"
        self.datetime_images = defaultdict(list)
        
        for day in os.listdir(f"{self.save_dir}/{self.tmp_dir}"):
            for hour in os.listdir(f"{self.save_dir}/{self.tmp_dir}/{day}"):
               directory = os.path.join(self.save_dir, self.tmp_dir, str(day), str(hour))
               for file in os.listdir(directory):
                   tif_file = TiffFile.parse(file)
                   if tif_file.band in self.input_layers:
                       self.datetime_images[tif_file.date].append(file)

        for box in self.boxes:
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


    def input_features(self, img_size:int = 32):
        for box in self.boxes:
            for time_stamp in os.listdir(os.path.join(self.save_dir, str(box.id))):
                parsed_date = datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                mask_file = os.path.join(self.save_dir, str(box.id), time_stamp, "output.tif")
                centre_x, centre_y = self.get_center_pixel(mask_file)

                # Finding offset by moving the center pixels to top-left according to image size
                x_random = int(random.uniform(-1 * (img_size // 3), img_size // 3))
                y_random = int(random.uniform(-1 * (img_size // 3), img_size // 3))
                x_offset = centre_y - img_size // 2 + x_random
                y_offset = centre_x - img_size // 2 + y_random

                window = (x_offset, y_offset, img_size, img_size)

                # Translate output.tif
                gdal.Translate(mask_file.replace("tif", "tiff"), mask_file, srcWin=window)
                os.remove(mask_file)


                date_from_year_start = (parsed_date - datetime(year=parsed_date.year, month=1, day=1))
                for file in self.datetime_images[parsed_date]:
                    src_file = os.path.join(self.save_dir, self.tmp_dir, str(date_from_year_start.days + 1), str(parsed_date.hour), file)
                    dst_file = os.path.join(self.save_dir, str(box.id), time_stamp, file)
                    gdal.Translate(dst_file, src_file, srcWin=window)


    def output_label(self):

        with open("./files/reprojected_NIFC_2023_Wildfire_Perimeters.json") as f:
            geojson_data = json.load(f)

        for box in self.boxes:
            for date in self.datetime_images.keys():
                if box.start < date and date < box.end:

                    multipolygon = self.create_polygon(geojson_data, box)
                    image_name = self.datetime_images[date][0]
                    date_from_year_start = (date - datetime(year=date.year, month=1, day=1)).days + 1
                    reference_raster_path = os.path.join(self.save_dir, self.tmp_dir, str(date_from_year_start), str(date.hour), image_name)

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


if __name__ == "__main__":
    ip = InputFeatures("./DATA")
    ip.output_label()
    ip.input_features()
