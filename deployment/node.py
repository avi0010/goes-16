import json
from typing import List
from osgeo import osr, gdal
import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


class Node:
    def __init__(
        self,
        device_id: str,
        id: str,
        device_name: str,
        device_type: str,
        lat: float,
        lon: float,
    ) -> None:
        self.device_id = device_id
        self.id = id
        self.device_name = device_name
        self.device_type = device_type
        self.lat = lat
        self.lon = lon


        base_dir = os.getenv("BASE_DATA_DIR")
        if base_dir is None:
            raise ValueError("BASE_DATA_DIR is not provided in env file")

        self.base_dir = base_dir

        if len(os.listdir(self.base_dir)) != 16 :
            raise ValueError("Base files not present")

        patches_dir = os.getenv("BASE_PATCHES_DIR")
        if patches_dir is None:
            raise ValueError("BASE_PATCHES_DIR is not provided in env file")

        self.patches_dir = patches_dir
        if not os.path.exists(self.patches_dir):
            os.mkdir(self.patches_dir)

        self.save_path = os.path.join(self.patches_dir, str(id))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.__convert_WSG__()

    def __convert_WSG__(self):
        InSR = osr.SpatialReference()
        InSR.SetFromUserInput("EPSG:4326")
        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        transform_epsg = osr.CoordinateTransformation(InSR, OutSR)

        self.lon, self.lat, _ = transform_epsg.TransformPoint(self.lat, self.lon)


    def crop(self, win_size:int=32):
        for file in os.listdir(self.base_dir):
            file_path = os.path.join(self.base_dir, file)
            layer = gdal.Open(file_path)

            transform = layer.GetGeoTransform()

            xOrigin = transform[0]
            yOrigin = transform[3]
            pixelWidth = transform[1]
            pixelHeight = -transform[5]

            col = (int((self.lon - xOrigin) / pixelWidth)) - win_size/2 - 1
            row = (int((yOrigin - self.lat) / pixelHeight)) - win_size/2 - 1

            window = (col, row, win_size, win_size)
            gdal.Translate(os.path.join(self.save_path, file), file_path, srcWin=window)

    @classmethod
    def parse(cls, node):
        device_id   = node["DeviceId"]
        id          = node["_id"]
        device_name = node["Device"]
        device_type = node["DeviceType"]
        lat         = node["lat"]
        lon         = node["lon"]

        return cls(device_id, id, device_name, device_type, lat, lon)


def node_isValid(node) -> bool:
    return isinstance(node["lat"], float) and isinstance(node["lon"], float)


def parse_json(file_path: str) -> List[Node]:
    nodes: List[Node] = []

    with open(file_path) as f:
        data = json.load(f)

        for node in data:
            if not node_isValid(node):
                continue

            nodes.append(Node.parse(node))

    return nodes


if __name__ == "__main__":
    nodes = parse_json("./deployment/locations.json")
    for node in tqdm(nodes):
        node.crop()
