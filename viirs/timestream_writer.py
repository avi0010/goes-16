import os
import json
import time
import math
import shutil
import uuid

import boto3

from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pyproj
from shapely import area
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
from geojson.utils import coords
import fiona
from shapely import to_wkt

load_dotenv()

aws_region = os.getenv('AWS_REGION')
aws_timestream_db = os.getenv('AWS_TIMESTREAM_DATABASE')
aws_timestream_table = 'n5-viirs-active-fire-product'

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
        {'Name':'lon', 'Value':str(coords[0]), 'Type':'VARCHAR'}, 
        {'Name':'lat', 'Value':str(coords[1]), 'Type':'VARCHAR'}
        ]
    record = {'Time': str(int(time.time() * 1000)), 'Dimensions': dimension, 'MeasureName': 'point_data', 'MeasureValueType': 'MULTI', 'MeasureValues':measure_values}
    return record

#WRITE A LIST OF RECORDS TO THE TIMESTREAM
def write_timestream(records, region, db, table): 
    
    timestream_write = boto3.client('timestream-write', region_name=region)
    try:
        timestream_write.write_records(DatabaseName=db, TableName=table, Records=records)
    except timestream_write.exceptions.RejectedRecordsException as err:
        print("RejectedRecords: ", err)
        for rr in err.response["RejectedRecords"]:
            print("Rejected Index " + str(rr["RecordIndex"]) + ": " + rr["Reason"])
    except Exception as err:
        print(f"Error: {err}")
    return

def chunker(input_list, chunk_size):
    """
    A generator function to yield fixed-size chunks from a given list.

    :param input_list: The list to be divided into chunks
    :param chunk_size: The size of each chunk
    :yield: A chunk of the list of size chunk_size
    """
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

if __name__ == '__main__':
    viirs_path = './files/viirs_data/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.shp'
    
    shapefile_data = fiona.open(viirs_path)

    records = []
    for feature in shapefile_data:
        acq_time = feature['properties']['ACQ_DATE'] + '_' + feature['properties']['ACQ_TIME']
        acq_time = datetime.strptime(acq_time, "%Y-%m-%d_%H%M")
        lat = feature['properties']['LATITUDE']
        lon = feature['properties']['LONGITUDE']
        id = uuid.uuid4().hex
        records.append(prepare_timestream_record(id, acq_time, [lon, lat]))

    print("Total records- ", len(records))
    for chunk in chunker(records, 100):
        write_timestream(chunk, aws_region, aws_timestream_db, aws_timestream_table)
    print("Done writing records")