import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Iterable, List

from pyproj import Geod
from shapely.geometry import shape, box
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

    @staticmethod
    def post_processing(geojson_dir:str, bbox, area_in_acres:float, base_dir):
        '''
            To filter out geometries that are outside bbox and are smaller than given area
        '''

        geojson_files = [os.path.join(geojson_dir, file) for file in os.listdir(geojson_dir)]

        geod = Geod(ellps="WGS84") # Specify a named ellipsoid
        bbox_polygon = box(*bbox)

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        output_dir = os.path.join(base_dir, "post-processed")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)


        for file in geojson_files:
            gj = json.load(open(file, 'r'))

            feature_collection = {
                'type': 'FeatureCollection',
                'features': []
            }

            for feature in gj['features']:
                geometry = shape(feature['geometry'])
                area = abs(geod.geometry_area_perimeter(geometry)[0]) * 0.000247105 # To convert to acres
                intersects = geometry.intersects(bbox_polygon)

                feature['properties']['area'] = area

                if intersects and area > area_in_acres:
                    feature_collection['features'].append(feature)
        
            output_path = os.path.join(output_dir, os.path.basename(file))
            #print(eligible, '\n')
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f)

    @staticmethod
    def transform_to_location_file(geojson_dir:str, save_dir):
        '''
        Takes the perimetes file and transform each polygon to centroid and save to separate files based on their timestamp
        '''

        geojson_files = [os.path.join(geojson_dir, file) for file in os.listdir(geojson_dir)]

        for file in geojson_files:

            gj = json.load(open(file, 'r'))

            feature_collection = {
                'type': 'FeatureCollection',
                'features': []
            }

            for feature in gj['features']:
                geometry = shape(feature['geometry'])        
                lon, lat = list(geometry.centroid.coords)[0]

                feature = {
                    "type": "Feature",
                    "properties": feature['properties'],
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                }

                feature_collection['features'].append(feature)

            locations_dir = os.path.join(save_dir, 'locations')
            if not os.path.exists(locations_dir):
                os.mkdir(locations_dir)
            output_path = os.path.join(locations_dir, os.path.basename(file))

            with open(output_path, 'w') as f:
                json.dump(feature_collection, f)

if __name__ == '__main__':
    shapefile = "./files/viirs_data/fire_nrt_J1V-C2_482482.shp"
    dataset = ViirsDatasetWriter(shapefile)

    dataset.save_by_date("data")

    # BBoxes- [-124.37,32.82,-94.84,48.79], [-94.67,24.5,-71.11,43.38]
    ViirsDatasetWriter.post_processing("data/output", [-94.67,24.5,-71.11,43.38], 70, "data")
    ViirsDatasetWriter.transform_to_location_file('data/post-processed', 'data/')