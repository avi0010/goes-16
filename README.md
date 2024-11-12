# GOES-16

## Downloading Latest

```
python3 goes-16/main.py -s /tmp/DATA/ latest
```

Will Download the latest bounding boxes in the save directory


## Bulk Downloading By datetime

```
python3 goes-16/main.py -s /tmp/DATA/ date -d 29-9-2023 1-10-2023
```
Will Download the specified date range bounding boxes in the save directory

## Bulk Downloading By bbox geojson

```bash
# First generate new bounding boxes with this script
$   python3 goes-16/bbox_generator_1.py [-g, --geojson-file] PATH_TO_JSON_FILE

# Now run download
$   python3 goes-16/main.py -s /tmp/DATA/ date [-g, --geojson]
```
Will Download the specified date range bounding boxes in the save directory

# Docker setup
## Building
```
docker build -t [image-name]:[image-tag] .
```
Builds the docker image

## Running
```
docker run  --rm  -v "/repo/path/goes-16:/app" goes_downloader:stable python3 goes-16/main.py -s DATA/ latest
```
Downloads latest images and saves them in `DATA` folder

```
docker run --rm  -v "/path/to/repo/goes-16:/app" [image-name]:[image-tag] python3 goes-16/main.py -s DATA/ date -d 2023-09-01 2023-10-02
```
Downloads images between start and end date, in above example 29-9-2023 & 1-10-2023, and saves them in `DATA` folder


```
docker run --network goes-16_default --rm  -v "/path/to/repo/goes-16:/app" goes_downloader:stable python3 goes-16/mosaic_update.py
```
Downloads images between start and end date, in above example 29-9-2023 & 1-10-2023, and saves them in `DATA` folder

```
docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/bbox_generator_1.py -f files/NIFC_2023_Wildfire_Perimeters.json
```
Generates bbox using json file

### Decoupled scripts
**Downloading cloud cover images**
```bash
# Download cloud images for entire US
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/DOWNLOAD_dated_bbox.py -s /app/DATA/ -p ABI-L2-ACMC

# Preprocess download cloud images (.nc to .tif conversion)
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/PREPROCESS_images_bbox.py -s /app/DATA/ -p ABI-L2-ACMC -b ACM
```

**Downloading and processing training images**
```bash

# Generate bounding boxes
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/new_bbox.py -f files/NIFC_2023_Wildfire_Perimeters.json

# Now download specific product images for entire US. In this example, we are taking ABI-L1b-RadC product
# Since this product will have 12 bands, filename will be prefixed with channel numbers, such as C04 or C12 etc.
# -d/--prev-days is optional and can be used to download past data as well
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/DOWNLOAD_dated_bbox.py -s /app/DATA/ -p ABI-L1b-RadC -d 15 -b 7,12,13,14,15

# Preprocess downloaded images, uses 'Rad' band (.nc to .tif conversion & cropping)
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/PREPROCESS_images_bbox.py -s /app/DATA/ -p ABI-L1b-RadC -b Rad -f radiance

# Input features
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 goes-16/input_features.py -d /app/DATA -p 15 -w 32
```

**Model training**
```bash
#training
# use flags --memory="2g" --memory-swap="5g" to use swap memory (should be present in system)

$	sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 training/train.py -d DATA -r 0.8 -e 50 -t 0.4 -m R2AttU

#validation
$	sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 training/val.py -d DATA/ -m training/models/R2AttU/model5_0.19709928333759308.pth
```


**New Downloading**
```bash
#Download patches
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 new_goes/download.py -s DATA -j files/Filtered_WFIGS_Interagency_Perimeters.json -p ABI-L1b-RadC
```

**Analysis**
```bash
#Download patches
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 new_goes/download_analytics.py

#Pixel Analysis
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 new_goes/pixel_analysis.py

#Colormap Analysis
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 new_goes/colormap_analysis.py
```

**Deployment**
```bash
#Download and Process files
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 deployment/download.py

#Generate patches
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 deployment/node.py

#Inference
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 deployment/infer.py

# Upload patches that have detection to S3 (only can be done from prod server)
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 deployment/upload_patches.py

#Writing to timestream database (only can be done from prod server)
$   sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 deployment/timestream_writer.py

#Viirs Validation
$   python3 viirs/val.py
$   sh run.sh
```
