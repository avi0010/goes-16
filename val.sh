d="2024/11/08 00:00:00"

while [ "$d" != "2024/11/10 00:00:00" ]; do 
  echo $d
  python3 ./deployment/download_string -t $d
  d=$(date -d "$d 1 hour" +"%Y/%m/%d %T")
  done
