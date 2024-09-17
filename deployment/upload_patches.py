import os
import json

from dotenv import load_dotenv
import shutil

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

load_dotenv()

def upload_file(file_name, bucket_name, object_name=None):

    s3_client = boto3.client('s3')

    # If S3 object_name is not specified, use file_name
    if object_name is None:
        object_name = file_name

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File '{file_name}' uploaded to '{bucket_name}/{object_name}'")
    except FileNotFoundError:
        print(f"The file '{file_name}' was not found.")
    except NoCredentialsError:
        print("Credentials not available.")
    except ClientError as e:
        print(f"Failed to upload file: {e}")

def check_file_format_for_upload(file:str) -> bool:
    FILE_FORMATS_NOT_ALLOWED_FOR_UPLOAD = ['.dbf', '.prj', '.shp', '.shx']

    for extension in FILE_FORMATS_NOT_ALLOWED_FOR_UPLOAD:
        if extension in file:
            return False
    
    return True

if __name__ == '__main__':

    bucket_name = os.getenv('AWS_BUCKET_NAME')

    hotspot_base_dir = os.getenv("BASE_PERIMETERS_DIR")
    hotspot_files = [os.path.join(hotspot_base_dir, file) for file in os.listdir(hotspot_base_dir) if file.find('json') > 0]

    if len(hotspot_files) > 1:
        raise Exception("Only one hotspot detection geojson should be present. Please delete data folder and re-run pipeline")
    

    _base_dir = os.getenv("BASE_DATA_DIR")

    _patch_dir = os.getenv("BASE_PATCHES_DIR")
    if _patch_dir is not None:
        patch_dir = _patch_dir
    else:
        raise ValueError("PATCH_DIR value not found")

    # Iterate over detections
    hotspot_file = hotspot_files[0]
    with open(hotspot_file, 'r') as f:
        hotspots = json.load(f)
    
        for feature in hotspots['features']:
            id = feature['properties']['id']
            timestamp = feature['properties']['timestamp'].replace(" ", "T")

            patch_folder = os.path.join(_patch_dir, id)
            patch_files = [os.path.join(patch_folder, file) for file in os.listdir(patch_folder) if check_file_format_for_upload(file)]
            
            for patch_file in patch_files:
                patch_file_name = os.path.basename(patch_file)
                obj_name = f"{timestamp}/{id}/{patch_file_name}"

                upload_file(patch_file, bucket_name, obj_name)

            print(f"Files uploaded to S3 for {id}")
    
    # cleanup
    shutil.rmtree(_base_dir)
    shutil.rmtree(_patch_dir)