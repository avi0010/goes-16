import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import json
# from osgeo import ogr


class Fire:
    def __init__(
        self,
        id: int,
        area: float,
        start: datetime,
        end: datetime,
#        geo: ogr.Geometry,
        fire_labels=None,
    ) -> None:
        self.id = id
        self.area_acre = area
        self.start_date = start
        self.end_date = end
#        self.geometry = geo
        self.fire = fire_labels

    def __str__(self) -> str:
        return f"{self.id}"

    @classmethod
    def parse(cls, fire):
        properties = fire["properties"]
        id = properties["poly_SourceOID"]
        area_acre = properties["poly_GISAcres"]

        fireDisoveryDateTime = properties["poly_PolygonDateTime"]

        fireControlDateTime = properties["attr_FireOutDateTime"]

        format_string = "%a, %d %b %Y %H:%M:%S %Z"
        start_date = datetime.strptime(fireDisoveryDateTime, format_string)
        end_date = datetime.strptime(fireControlDateTime, format_string)
#        geometry = ogr.CreateGeometryFromJson(json.dumps(fire["geometry"]))

        fires = None
        if "fire_samples" in properties:
            fires = properties["fire_samples"]

        return cls(id, area_acre, start_date, end_date, fires)


def read_json_file(path: str) -> List[Fire]:
    fires: List[Fire] = []
    with open(path) as f:
        data = json.load(f)
        for fire_data in data["features"]:
            area = fire_data["properties"]["poly_GISAcres"]
            start = fire_data["properties"]["poly_PolygonDateTime"]
            end = fire_data["properties"]["attr_FireOutDateTime"]
            if area < 10.0 or start is None or end is None:
                continue
            fire = Fire.parse(fire_data)
            fires.append(fire)
    return fires


def parse_filename(filename: str) -> dict:
    if filename.startswith("OR_"):
        filename = filename[3:]

    parts = filename.split("_")
    if len(parts) != 5:
        raise ValueError("Invalid filename format")

    channel = None

    band_data = parts[0].split("-")
    product = band_data[2]
    if product == "RadC":
        channel = int(parts[0][-2:])

    start_time = parts[2][1:]
    end_time = parts[3][1:]

    start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")
    end_dt = datetime.strptime(end_time, "%Y%j%H%M%S%f")

    return {
        "channel": channel,
        "product": product,
        "start_time": start_dt,
        "end_time": end_dt,
    }


class ModelInput:
    def __init__(self, in_dir, pos) -> None:
        self.in_dir = in_dir
        self.pos = pos

    def create_input(self):
        for band in os.listdir(self.in_dir):
            if ".tif" not in band:
                continue

            if "output" in band:
                if self.pos:
                    output = Image.open(os.path.join(self.in_dir, band))
                else:
                    output = torch.zeros((1, 32, 32))
                continue

            f = parse_filename(band)

            if f["channel"] == 1:
                band_1 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_1 = (band_1 - 0.0) / (1 - 0)

            elif f["channel"] == 2:
                band_2 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_2 = (band_2 - 0.0) / (1 - 0)

            elif f["channel"] == 3:
                band_3 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_3 = (band_3 - 0.0) / (1 - 0)

            elif f["channel"] == 4:
                band_4 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_4 = (band_4 - 0.0) / (1 - 0)

            elif f["channel"] == 5:
                band_5 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_5 = (band_5 - 0.0) / (1 - 0)

            elif f["channel"] == 6:
                band_6 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_6 = (band_6 - 0.0) / (1 - 0)

            elif f["channel"] == 7:
                mn, mm = 197.31, 411.86
                band_7 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_7 = (band_7 - mn) / (mm - mn)

            elif f["channel"] == 8:
                mn, mm = 138.05, 311.06
                band_8 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_8 = (band_8 - mn) / (mm - mn)

            elif f["channel"] == 9:
                mn, mm = 137.7, 311.08
                band_9 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_9 = (band_9 - mn) / (mm - mn)

            elif f["channel"] == 10:
                mn, mm = 126.91, 331.2
                band_10 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_10 = (band_10 - mn) / (mm - mn)

            elif f["channel"] == 11:
                mn, mm = 127.69, 341.3
                band_11 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_11 = (band_11 - mn) / (mm - mn)

            elif f["channel"] == 12:
                mn, mm = 117.49, 311.06
                band_12 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_12 = (band_12 - mn) / (mm - mn)

            elif f["channel"] == 13:
                mn, mm = 89.62, 341.28
                band_13 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_13 = (band_13 - mn) / (mm - mn)

            elif f["channel"] == 14:
                mn, mm = 96.19, 341.28
                band_14 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_14 = (band_14 - mn) / (mm - mn)

            elif f["channel"] == 15:
                mn, mm = 97.38, 341.28
                band_15 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_15 = (band_15 - mn) / (mm - mn)

            elif f["channel"] == 16:
                mn, mm = 92.7, 318.26
                band_16 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_16 = (band_16 - mn) / (mm - mn)

        inputs = torch.concatenate(
            (
                band_1,
                band_2,
                band_3,
                band_4,
                band_5,
                band_6,
                band_7,
                band_8,
                band_9,
                band_10,
                band_11,
                band_12,
                band_13,
                band_14,
                band_15,
                band_16,
            )
        )

        return inputs, output


class CustomDataset(Dataset):
    def __init__(
        self, datalist: List[ModelInput], transforms=None, target_transforms=None
    ) -> None:
        super().__init__()
        self.datalist = datalist
        self.transform = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        """
        The __len__ function returns the number of samples in dataset
        """
        return len(self.datalist)

    def __getitem__(self, index):
        """
        The __getitem__ function loads and returns a sample from the dataset at the given index idx
        """
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        input_images, label = self.datalist[index].create_input()

        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            input_images = self.transform(input_images)

        if self.target_transforms is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transforms(label)

        return input_images, label
