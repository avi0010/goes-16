#!/bin/bash

#Download and Process files
sudo docker run --rm  -v "/mnt/dev/goes-16:/app" goes_downloader:stable python3 deployment/download.py

#Generate patches
sudo docker run --rm  -v "/mnt/dev/goes-16:/app" goes_downloader:stable python3 deployment/node.py

#Inference
sudo docker run --rm  -v "/mnt/dev/goes-16:/app" goes_downloader:stable python3 deployment/infer.py

# Upload patches that have detection to S3 (only can be done from prod server)
sudo docker run --rm  -v "/mnt/dev/goes-16:/app" goes_downloader:stable python3 deployment/upload_patches.py

#Writing to timestream database (only can be done from prod server)
sudo docker run --rm  -v "/mnt/dev/goes-16:/app" goes_downloader:stable python3 deployment/timestream_writer.py