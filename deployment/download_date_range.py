from download import Downloader
from node import Node
from datetime import datetime, timedelta
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", required=True)
    parser.add_argument("-e", "--end", required=True)
    parser.add_argument("-d", "--diff", required=True)

    args = parser.parse_args()

    START_TIME  = datetime.strptime(args.start, "%d/%m/%Y-%H:%M:%S")
    END_TIME = datetime.strptime(args.end, "%d/%m/%Y-%H:%M:%S")

    dates = [START_TIME + timedelta(minutes=x*float(args.diff)) for x in range(int((END_TIME - START_TIME).total_seconds()/(60 * float(args.diff)) + 1))]

    coordinates = [[39.10353899002075, -77.06602717886126]]

    down = Downloader()
    for idx, i in enumerate(dates):
        down.download_datetime(i)

    for i, coord in enumerate(coordinates):
        node = Node(device_id=str(i), id=str(i), device_name=str(i), device_type=str(i), lat=coord[0], lon=coord[1])
        node.crop()
