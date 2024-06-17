import os
import json
import time
import math
import shutil

import boto3

from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pyproj
from shapely import area
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
from geojson.utils import coords

load_dotenv()

def reproject_geojson(geojson_path, src_crs, dst_crs, output_path):

    def get_coords_from_polygon(shape):
        coords = set()  

        if shape.geom_type == 'Polygon':
            coords.update(shape.exterior.coords[:-1])
            for linearring in shape.interiors:
                coords.update(linearring.coords[:-1])
        elif shape.geom_type == 'MultiPolygon':
            for polygon in shape.geoms:
                coords.update(get_coords_from_polygon(polygon)) 

        return coords


    """
    Reproject a GeoJSON file from the source CRS to the destination CRS.

    Parameters:
    - geojson_path: Path to the input GeoJSON file.
    - src_crs: Source coordinate reference system (CRS) string.
    - dst_crs: Destination coordinate reference system (CRS) string.
    - output_path: Path to save the reprojected GeoJSON file.

    Returns:
    - None
    """
    # Load GeoJSON data
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Define projection transformers
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    reverse_project = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True).transform

    feature_layer = {
        "type": "FeatureCollection",
        "name": "Perimeters",
        "crs": {
            "type": "name",
            "properties": {
            }
        },
        "features": []
    }

    # Reproject each feature geometry
    for feature in geojson_data['features']:
        print(feature)

        geom = shape(feature['geometry'])
        reprojected_geom = transform(project, geom)
        print(reprojected_geom)

        reprojected_feature = {
            'type': 'feature',
            'properties': feature['properties'],
            'geometry': mapping(reprojected_geom)
        }

        geom = shape(reprojected_feature['geometry'])
        
        coordinates = get_coords_from_polygon(geom)
        invalid_coordinates = False
        for coordinate in coordinates:
            if math.isinf(coordinate[0]) or math.isinf(coordinate[1]):
                invalid_coordinates = True
                break
        if invalid_coordinates:
            continue                

        feature_layer['features'].append(reprojected_feature)

    # Write reprojected GeoJSON to file
    with open(output_path, 'w') as f:
        json.dump(feature_layer, f)

#PREPARE SINGLE TIMESTREAM RECORD
#Change function arguments to contain whatever data you want in your timesream
def prepare_timestream_record(stationID, current_time, coords):
    #Set dimensional measures
    dimension = [ 
        {'Name': 'stationID', 'Value': stationID}, 
        {'Name': 'PRIMARY_KEY', 'Value': str(current_time)}, 
        ] 

    #Set multi-measures - whatever values you need in your timestream entry
    measure_values = [
        #{'Name':'code', 'Value':str(code), 'Type':'BIGINT'}, 
        #{'Name':'int_level', 'Value':str(level), 'Type':'BIGINT'}, 
        {'Name':'lon', 'Value':json.dumps(list(np.array(coords)[:,0])), 'Type':'VARCHAR'}, 
        {'Name':'lat', 'Value':json.dumps(list(np.array(coords)[:,1])), 'Type':'VARCHAR'}
        ]
    record = {'Time': str(int(time.time() * 1000)), 'Dimensions': dimension, 'MeasureName': 'polygon_data', 'MeasureValueType': 'MULTI', 'MeasureValues':measure_values}
    return record

#WRITE A LIST OF RECORDS TO THE TIMESTREAM
def write_timestream(records, db='n5-timestream', table='hotspot-perimeters'): 
    #boto3.client('timestream-write').write_records(
    #DatabaseName=db, TableName=table,
    #Records = records
    #)
    
    timestream_write = boto3.client('timestream-write', region_name='us-east-2')
    try:
        timestream_write.write_records(DatabaseName=db, TableName=table, Records=records)
    except timestream_write.exceptions.RejectedRecordsException as err:
        print("RejectedRecords: ", err)
        for rr in err.response["RejectedRecords"]:
            print("Rejected Index " + str(rr["RecordIndex"]) + ": " + rr["Reason"])
    except Exception as err:
        print(f"Error: {err}")
    return

if __name__ == '__main__':

    base_dir = os.getenv("BASE_DIR")
    hotspot_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.find('json') > 0]

    records_to_write = []
    for gj in hotspot_files:

        # reprojecting the geojson file from 102498 to 4326
        reproject_geojson(gj, 'ESRI:102498', 'EPSG:4326', gj)

        # load GeoJSON data
        with open(gj, 'r') as f:
            geojson_data = json.load(f)

        # convert goes img scan time str to unix timestamp
        acquisition_time_str = os.path.basename(gj).replace('hotspots_', '').replace('.json', '')
        acquisition_timestamp = int(datetime.strptime(acquisition_time_str, '%Y-%m-%d %H:%M:%S.%f').timestamp() * 1000)

        for feature in geojson_data['features']:

            #append any data item(s) to records_to_write
            id = feature['properties']['id']
            polygon_coords = list(np.array(list(coords(feature))))

            records_to_write.append(prepare_timestream_record(stationID=id, current_time=acquisition_timestamp, coords=polygon_coords))      

    print(f'Writing {len(records_to_write)} records')

    #Write to timestream
    write_timestream(records_to_write)

    # Clean BASE DIR
    shutil.rmtree(base_dir)