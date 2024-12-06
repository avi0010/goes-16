start="2024-11-08 00:00:00"
end="2024-11-10 00:00:00"
mkdir validation_data
while [ "$start" != "$end" ]; do 
  echo $start
  sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 ./deployment/download_string -t $start
  sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 ./deployment/node.py
  sudo docker run --rm  -v ".:/app" goes_downloader:stable python3 ./deployment/infer.py
  mv $BASE_PATHES_DIR validation_data/$start
  rm -rf $BASE_DIR
  start=$(date -d "$start 1 hour" +"%Y-%m-%d %T")
  done
