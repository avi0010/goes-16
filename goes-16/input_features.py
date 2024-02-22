import numpy as np
from osgeo import ogr, gdal, osr
from PIL import Image
from datetime import datetime, timedelta
from bbox import Bboxs
import s3fs
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
            band = np.where(band < np.percentile(band, 99.8), band, 0)
            arr.append(band.tolist())
            os.remove(os.path.join(self.save_dir_cropped, f))

        result = np.array(arr).mean(axis=0)
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
        self.layers=[[7],
                     [7, 12],
                     [7, 13],
                     [7, 14],
                     [7, 15]]

        self.save_dir = save_dir
        self.days = [int(i) for i in os.listdir(f"{save_dir}") if i != "tmp"]
        self.features = {}
        for box in self.days:
            self.features[box] = os.listdir(f"{save_dir}/{box}/")[0]

    def get_file(self, file_list: list[str], band_id):
        for file in file_list:
           f = TiffFile.parse(file)
           if f.band == band_id:
               return file

    def parse_datetime(self, date):
        # 2024-01-03T17:39:21Z
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])
        hour = int(date[11:13])
        minutes = int(date[14:16])
        sec = int(date[17:19])

        return datetime(year=year, month=month, day=day, hour=hour, minute=minutes, second=sec, microsecond=0)

    def save_output(self, box, image_path, date):
        
        with open("./files/NIFC_2023_Wildfire_Perimeters.json") as f:
            geojson_data = json.load(f)

        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for poly in geojson_data["features"]:
            if poly["properties"]["poly_SourceOID"] == int(box):
                geom = ogr.CreateGeometryFromJson(json.dumps(poly['geometry']))
                if geom.GetGeometryName() == 'MULTIPOLYGON':
                    for i in range(geom.GetGeometryCount()):
                        multipolygon.AddGeometry(geom.GetGeometryRef(i))
                elif geom.GetGeometryName() == 'POLYGON':
                    multipolygon.AddGeometry(geom)

        raster_layer = gdal.Open(image_path)
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
        InSR.SetFromUserInput("EPSG:4326")
        mem_layer = mem_ds.CreateLayer('multipolygon', geom_type=ogr.wkbMultiPolygon, srs=InSR)
        feature_defn = mem_layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        feature.SetGeometry(multipolygon)
        mem_layer.CreateFeature(feature)

        gdal.RasterizeLayer(target_layer, [1], mem_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])

        output_file = f"{self.save_dir}/{box}/{str(date)}/output.tif"
        gdal.Translate(output_file, target_layer, format='GTiff')
        return
           
    def get_features(self):
        for box, feature in self.features.items():
            with open("./files/NIFC_2023_Wildfire_Perimeters.json") as f:
                geojson_data = json.load(f)

            for poly in geojson_data["features"]:
                if poly["properties"]["poly_SourceOID"] == int(box):
                    properties = poly["properties"]
                    fireDisoveryDateTime = properties['attr_FireDiscoveryDateTime'] if properties['attr_FireDiscoveryDateTime'] is not None else properties['poly_CreateDate']
                    fireControlDateTime = properties['attr_ContainmentDateTime'] if properties['attr_ContainmentDateTime'] is not None else properties['attr_ModifiedOnDateTime_dt']

                    fireDisoveryDateTime = self.parse_datetime(fireDisoveryDateTime)
                    fireControlDateTime = self.parse_datetime(fireControlDateTime)
            
            time_file_dict = {}
            for file in os.listdir(f"{self.save_dir}/{box}/{feature}"):
               f = TiffFile.parse(file)
               time_file_dict[f.date] = []

            for file in os.listdir(f"{self.save_dir}/{box}/{feature}"):
                f = TiffFile.parse(file)
                time_file_dict[f.date].append(f.file_name)

            for date, files in time_file_dict.items():
                if date < fireDisoveryDateTime or date > fireControlDateTime:
                    return
                os.mkdir(f"{self.save_dir}/{box}/{str(date)}")
                past= PastFeatures(self.save_dir, box, date)
                past.process()
                for i, layer in enumerate(self.layers):
                    base_dir = f"{self.save_dir}/{box}/{feature}/"
                    if len(layer) == 1:
                        f = self.get_file(files, layer[0])
                        image_path = os.path.join(base_dir, f)
                        self.save_output(box, image_path, date)
                        img = gdal.Open(image_path)
                        band = np.array(img.GetRasterBand(1).ReadAsArray())
                        band = np.where(band < np.percentile(band, 99.8), band, 0)
                        im = Image.fromarray(band)
                        im = im.resize((64, 64))
                        im.save(f"{self.save_dir}/{box}/{str(date)}/b_{i}.tiff")

                    else:
                        f1 = self.get_file(files, layer[0])
                        img1 = gdal.Open(os.path.join(base_dir, f1))
                        band1 = np.array(img1.GetRasterBand(1).ReadAsArray())
                        band1 = np.where(band1 < np.percentile(band1, 99.8), band1, 0)


                        f2 = self.get_file(files, layer[1])
                        img2 = gdal.Open(os.path.join(base_dir, f2))
                        band2 = np.array(img2.GetRasterBand(1).ReadAsArray())
                        band2 = np.where(band2 < np.percentile(band2, 99.8), band2, 0)

                        diff = band1 - band2 
                        im = Image.fromarray(diff)
                        im = im.resize((64, 64))
                        im.save(f"{self.save_dir}/{box}/{str(date)}/b_{i}.tiff")


if __name__ == "__main__":
    ip = InputFeatures("./DATA")
    ip.get_features()
