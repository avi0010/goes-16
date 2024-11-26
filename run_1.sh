#!/bin/bash

TARGET_DIR="./ttttt/patches/"
HOTSPOTS_DIR="./ttttt/hotspots/"

# List all directories inside the TARGET_DIR and store them in a variable
mapfile -t DIRS < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d)

for dir in "${DIRS[@]}"; do
	#find "$dir" -type f -regex ".*/output_[0-9]+\.tiff" -delete
	#export BASE_PATCHES_DIR="$dir"
	timestamp=$(basename "$dir")
	echo "$dir"
    sudo docker run --rm -e BASE_PATCHES_DIR="$dir" -e TIMESTAMP="$timestamp" -e BASE_PERIMETERS_DIR="$HOTSPOTS_DIR" -v ".:/app" goes_downloader:stable python3 deployment/infer_1.py
done
