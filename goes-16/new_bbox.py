import os
import json
import numpy as np
import argparse
import random
from geojson import Polygon, Feature
from shapely.geometry import shape

def generate_square_crop_geojson(center_point, startDate, endDate, fireArea):
    box_size_km = (16384) ** 0.5
    center_latitude = (center_point[1] + center_point[3])/2
    center_longitude = (center_point[0] + center_point[2])/2

    # Calculate the coordinates of the square corners
    min_latitude = center_latitude - (box_size_km / 2) / 111.111
    max_latitude = center_latitude + (box_size_km / 2) / 111.111
    min_longitude = center_longitude - (box_size_km / 2) / (111.321 * abs(np.cos(np.radians(center_latitude))))
    max_longitude = center_longitude + (box_size_km / 2) / (111.321 * abs(np.cos(np.radians(center_latitude))))

    # Generate random offsets for latitude and longitude
    lat_offset = abs((max_latitude - min_latitude) / 3)
    latitude_offset = random.uniform(-1 * lat_offset, lat_offset)  # Adjust the range as needed
    long_offset = abs((max_longitude - min_longitude) / 3)
    longitude_offset = random.uniform(-1 * long_offset, long_offset)  # Adjust the range as needed

    min_latitude = center_latitude - (box_size_km / 2) / 111.111 + latitude_offset
    max_latitude = center_latitude + (box_size_km / 2) / 111.111 + latitude_offset
    min_longitude = center_longitude - (box_size_km / 2) / (111.321 * abs(np.cos(np.radians(center_latitude)))) + longitude_offset
    max_longitude = center_longitude + (box_size_km / 2) / (111.321 * abs(np.cos(np.radians(center_latitude)))) + longitude_offset

    # Create GeoJSON Polygon
    polygon_coordinates = [
        (min_longitude, max_latitude),
        (min_longitude, min_latitude),
        (max_longitude, min_latitude),
        (max_longitude, max_latitude),
        (min_longitude, max_latitude)]

    polygon=Polygon()
    polygon['coordinates']=[polygon_coordinates]
    feature = Feature(geometry=polygon)
    feature.properties['start_date'] = startDate
    feature.properties['end_date'] = endDate
    feature.properties['area'] = fireArea

    finalJson ={
        "type":"FeatureCollection",
        "features":[feature]
    }
    return finalJson


def dumpjson(root:str, bounding_box_geojson, filename:str):
    json_dump_dir = os.path.join(os.path.dirname(root), 'geojson')
    filename+=".json"
    with open(os.path.join(json_dump_dir,filename),'w') as json_file:
        json.dump(bounding_box_geojson, json_file,indent=2)

def geojson(file: str):
    if file.endswith(".json") == False:
        return

    with open(file) as geojson_file:
        geojson_dict = json.load(geojson_file)
        features = geojson_dict["features"]

        for feature in features:
            geometry = feature["geometry"]
            properties = feature["properties"]
            shapely_geometry = shape(geometry)
            sourceOID = properties["poly_SourceOID"]
            area_acres = properties['poly_GISAcres']

            fireDisoveryDateTime = properties['attr_FireDiscoveryDateTime'] if properties['attr_FireDiscoveryDateTime'] is not None else properties['poly_CreateDate']
            fireControlDateTime = properties['attr_ContainmentDateTime'] if properties['attr_ContainmentDateTime'] is not None else properties['attr_ModifiedOnDateTime_dt']


            if area_acres < 10.0 or (fireDisoveryDateTime is None or fireControlDateTime is None):
                continue

            bounding_box = shapely_geometry.bounds
            bounding_box_geojson = generate_square_crop_geojson(bounding_box, fireDisoveryDateTime, fireControlDateTime, area_acres)

            root = os.path.dirname(os.path.abspath(__file__))
            dumpjson(root, bounding_box_geojson,str(sourceOID))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)

    args = parser.parse_args()
    file_path = args.file

    geojson(file_path)

if __name__ == '__main__':
    main()
