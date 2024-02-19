import numpy as np
from osgeo import ogr, gdal, osr
from PIL import Image
from datetime import datetime
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
                properties = poly["properties"]
                fireDisoveryDateTime = properties['attr_FireDiscoveryDateTime'] if properties['attr_FireDiscoveryDateTime'] is not None else properties['poly_CreateDate']
                fireControlDateTime = properties['attr_ContainmentDateTime'] if properties['attr_ContainmentDateTime'] is not None else properties['attr_ModifiedOnDateTime_dt']

                fireDisoveryDateTime = self.parse_datetime(fireDisoveryDateTime)
                fireControlDateTime = self.parse_datetime(fireControlDateTime)

                if date > fireDisoveryDateTime and date < fireControlDateTime:
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
            time_file_dict = {}
            for file in os.listdir(f"{self.save_dir}/{box}/{feature}"):
               f = TiffFile.parse(file)
               time_file_dict[f.date] = []

            for file in os.listdir(f"{self.save_dir}/{box}/{feature}"):
                f = TiffFile.parse(file)
                time_file_dict[f.date].append(f.file_name)

            for date, files in time_file_dict.items():
                os.mkdir(f"{self.save_dir}/{box}/{str(date)}")
                for i, layer in enumerate(self.layers):
                    base_dir = f"{self.save_dir}/{box}/{feature}/"
                    if len(layer) == 1:
                        f = self.get_file(files, 3)
                        image_path = os.path.join(base_dir, f)
                        self.save_output(box, image_path, date)
                        im = Image.open(image_path)
                        im = im.resize((128, 128))
                        imarray = np.array(im)
                        im = Image.fromarray(imarray)
                        im.save(f"{self.save_dir}/{box}/{str(date)}/b_{i}.tiff")

                    else:
                        f1 = self.get_file(files, layer[0])
                        im1 = Image.open(os.path.join(base_dir, f1))
                        im1 = im1.resize((128, 128))
                        imarray1 = np.array(im1)


                        f2 = self.get_file(files, layer[1])
                        im2 = Image.open(os.path.join(base_dir, f2))
                        im2 = im2.resize((128, 128))
                        imarray2 = np.array(im2)

                        diff = imarray1 - imarray2
                        im = Image.fromarray(diff)
                        im.save(f"{self.save_dir}/{box}/{str(date)}/b_{i}.tiff")


if __name__ == "__main__":
    ip = InputFeatures("./DATA")
    ip.get_features()
