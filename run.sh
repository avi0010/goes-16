TARGET_DIR="./ttttt/patches/"

# List all directories inside the TARGET_DIR and store them in a variable
mapfile -t DIRS < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d)

for dir in "${DIRS[@]}"; do
	echo "$dir"
	find "$dir" -type f -regex ".*/output_[0-9]+\.tiff" -delete
	export BASE_PATCHES_DIR="$dir"
    python3 deployment/infer.py
done
