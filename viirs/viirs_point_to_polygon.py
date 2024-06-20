import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
from pprint import pprint
from osgeo import gdal, ogr, osr
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from viirs import ViirsPoint, ViirsDataset

class ViirsDatasetWriter(ViirsDataset):

    def process_output(self):

        mem_driver = ogr.GetDriverByName("Memory")
        mem_ds = mem_driver.CreateDataSource("mem_data_source")
        InSR = osr.SpatialReference()
        InSR.SetFromUserInput("EPSG:4326")
        mem_layer = mem_ds.CreateLayer("multipolygon", geom_type=ogr.wkbMultiPolygon, srs=InSR)
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        ds = defaultdict(list)
        for idx, p in enumerate(self._db.labels_):
            ds[p].append(idx)

        self.__polygons = []
        for k, v in ds.items():
            if k == -1:
                continue

            polygon_points = [[self.filtered_data_points[x].lon, self.filtered_data_points[x].lat] for x in v]

            try:
                if len(polygon_points) > 0:
                    hull = ConvexHull(polygon_points)
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for p in hull.vertices:
                        lon = polygon_points[p][0]
                        lat = polygon_points[p][1]
                        ring.AddPoint(lon, lat)

                    # Complete ring
                    if len(hull.vertices) > 0:
                        lon = polygon_points[hull.vertices[0]][0]
                        lat = polygon_points[hull.vertices[0]][1]
                        ring.AddPoint(lon, lat)

                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                    poly.FlattenTo2D() # Converting to 2D polygon
                    self.__polygons.append(poly)  
            except Exception as e:
                print("Error- ", e)
                print("Skipping polygon")

    def save_by_date(self, base_dir:str):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.output_dir = os.path.join(self.base_dir, "output")
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        polygons_by_date = dict() #DD-MM-YYY
        for date in self.unique_dates:
            self.fit(date)
            self.process_output()
            
            date_str = date.strftime("%d-%m-%Y")
            if not date_str in polygons_by_date:
                polygons_by_date[date_str] = []
            
            features = [{"type": "Feature", "properties": {"timestamp": date.strftime("%d/%m/%Y %H:%M:%S")}, "geometry": json.loads(feature.ExportToJson())}
                                                    for feature in self.__polygons]
            polygons_by_date[date_str].extend(features)

        for date_str in polygons_by_date:
            output_path = os.path.join(self.output_dir, date_str + '.json')

            feature_collection = {
                'type': 'FeatureCollection',
                'features': polygons_by_date[date_str]
            }

            with open(output_path, 'w') as f:
                json.dump(feature_collection, f)

    def save(self, base_dir:str):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.output_dir = os.path.join(self.base_dir, "output")
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        for date in self.unique_dates:
            self.fit(date)
            self.process_output()
            
            output_path = os.path.join(self.output_dir, date.strftime("%d-%m-%YT%H-%M-%S") + '.json')
            feature_collection = {
                                    'type': 'FeatureCollection',
                                    'features': [{"type": "Feature", "properties": {"timestamp": date.strftime("%d/%m/%Y %H:%M:%S")}, "geometry": json.loads(feature.ExportToJson())}
                                                    for feature in self.__polygons]
                            }
        
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f)

if __name__ == '__main__':
    shapefile = "./files/viirs_data/fire_nrt_J1V-C2_441088.shp"
    dataset = ViirsDatasetWriter(shapefile)
    #dataset.save("data")
    dataset.save_by_date("data")