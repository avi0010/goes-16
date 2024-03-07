import argparse
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class TiffFile:
    def __init__(self) -> None:
        self.band: int
        self.date: datetime
        self.file_name: str

    def __str__(self) -> str:
        return f"{self.band, self.date, self.file_name}"

    @classmethod
    def parse(cls, name: str):
        cls.file_name = name
        split = name.split("_")
        cls.band = int(split[0][1:])
        year = split[1][0:4]
        month = split[1][4:6]
        day = split[1][6:8]
        hour = split[1][9:11]
        minute = split[1][11:13]
        seconds = split[1][13:15]
        microseconds = split[1][15:21]
        date_time = datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
            second=int(seconds),
            microsecond=int(microseconds),
        )
        cls.date = date_time
        return cls


class ModelInput:
    def __init__(self, in_dir) -> None:
        self.inputs = []
        self.in_dir = in_dir
        for band in os.listdir(in_dir):
            if ".tif" not in band:
                continue
            if "output" in band:
                self.out = os.path.join(in_dir, band)
                continue

            self.inputs.append(os.path.join(in_dir, band))
            if "band_5" in band:
                self.temporal_band = os.path.join(in_dir, band)
                self.temporal_range = [197.31, 411.86]
                continue

            file = TiffFile.parse(band)
            if file.band == 7:
                self.band_7 = os.path.join(in_dir, band)
                self.range_7 = [197.31, 411.86]
            elif file.band == 12:
                self.band_12 = os.path.join(in_dir, band)
                self.range_12 = [117.49, 311.06]
            elif file.band == 13:
                self.band_13 = os.path.join(in_dir, band)
                self.range_13 = [89.62, 341.27]
            elif file.band == 14:
                self.band_14 = os.path.join(in_dir, band)
                self.range_14 = [96.19, 341.28]
            elif file.band == 15:
                self.band_15 = os.path.join(in_dir, band)
                self.range_15 = [97.38, 341.28]

    def __str__(self) -> str:
        return f"[{self.band_7}, {self.band_12}, {self.band_13}, {self.band_14}, {self.band_15}, {self.temporal_band}] -> {self.out}"


class CustomDataset(Dataset):
    def __init__(
        self, data_list: List[ModelInput], transforms=None, target_transforms=None
    ) -> None:
        """
        The __init__ function is run once when instantiating the Dataset object
        """
        super().__init__()
        self.data_list = data_list
        self.transform = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        """
        The __len__ function returns the number of samples in dataset
        """
        return len(self.data_list)

    def __getitem__(self, index):
        """
        The __getitem__ function loads and returns a sample from the dataset at the given index idx
        """
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        if self.transform is not None:
            inputs = self.data_list[index]

            band_7 = torch.clamp(
                v2.ToImage()(Image.open(inputs.band_7)),
                min=inputs.range_7[0],
                max=inputs.range_7[1],
            )
            band_7 = (band_7 - inputs.range_7[0]) / (
                inputs.range_7[1] - inputs.range_7[0]
            )
            band_12 = torch.clamp(
                v2.ToImage()(Image.open(inputs.band_12)),
                min=inputs.range_12[0],
                max=inputs.range_12[1],
            )
            band_12 = (band_12 - inputs.range_12[0]) / (
                inputs.range_12[1] - inputs.range_12[0]
            )

            band_13 = torch.clamp(
                v2.ToImage()(Image.open(inputs.band_13)),
                min=inputs.range_13[0],
                max=inputs.range_13[1],
            )
            band_13 = (band_13 - inputs.range_13[0]) / (
                inputs.range_13[1] - inputs.range_13[0]
            )

            band_14 = torch.clamp(
                v2.ToImage()(Image.open(inputs.band_14)),
                min=inputs.range_13[0],
                max=inputs.range_13[1],
            )
            band_14 = (band_14 - inputs.range_14[0]) / (
                inputs.range_14[1] - inputs.range_14[0]
            )

            band_15 = torch.clamp(
                v2.ToImage()(Image.open(inputs.band_15)),
                min=inputs.range_15[0],
                max=inputs.range_15[1],
            )
            band_15 = (band_15 - inputs.range_15[0]) / (
                inputs.range_15[1] - inputs.range_15[0]
            )

            temporal_band = torch.clamp(
                v2.ToImage()(Image.open(inputs.temporal_band)),
                min=inputs.temporal_range[0],
                max=inputs.temporal_range[1],
            )
            temporal_band = (temporal_band - inputs.temporal_range[0]) / (
                inputs.temporal_range[1] - inputs.temporal_range[0]
            )

            input_images = torch.concatenate(
                (
                    band_7,
                    band_12 - band_7,
                    band_13 - band_7,
                    band_14 - band_7,
                    band_15 - band_7,
                    temporal_band - band_7,
                )
            )

            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            input_images = self.transform(input_images)

        label = Image.open(self.data_list[index].out)

        if self.target_transforms is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transforms(label)

        return input_images, label
