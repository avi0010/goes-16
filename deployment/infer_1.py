import os
import torch
import random
from collections import defaultdict
import json
import numpy as np
from typing import List
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from osgeo import gdal, ogr, osr
from dotenv import load_dotenv
import analysis_utils
import matplotlib.pyplot as plt
import shutil

load_dotenv()
cm = plt.get_cmap("viridis")

def rgb_image(dict, path):
    gamma = 2.2
    B = np.array(Image.open(os.path.join(path, dict[1])))
    R = np.array(Image.open(os.path.join(path, dict[2])))
    G = np.array(Image.open(os.path.join(path, dict[3])))

    B = np.power(B, 1/gamma)
    G = np.power(G, 1/gamma)
    R = np.power(R, 1/gamma)

    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    RGB = (np.dstack([R, G_true, B]) * 255).astype(np.uint8)
    return Image.fromarray(RGB)

def parse_filename(filename: str) -> dict:
    if filename.startswith("OR_"):
        filename = filename[3:]

    parts = filename.split("_")
    if len(parts) > 6:
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
    def __init__(self, in_dir) -> None:
        if isinstance(in_dir, str):
            self.files = [os.path.join(in_dir, fire) for fire in os.listdir(in_dir)]
            self.in_dir = in_dir
        elif isinstance(in_dir, list):
            self.files = in_dir
            self.in_dir = "/".join(in_dir[0].split("/")[:-1])
        else:
            raise ValueError("Invalid Input ")

    def create_input(self):
        for band in self.files:
            if ".tif" not in band:
                continue

            f = parse_filename(band.split("/")[-1])

            if f["channel"] == 1:
                band_1 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=0,
                    max=1,
                )
                band_1 = (band_1 - 0.0) / (1 - 0)

            elif f["channel"] == 2:
                raster_layer = gdal.Open(band)

                driver = gdal.GetDriverByName("Gtiff")
                output_dataset = driver.Create(
                    os.path.join(self.in_dir, f"{f['start_time']}.tiff"),
                    raster_layer.RasterXSize,
                    raster_layer.RasterYSize,
                    1,
                    gdal.GDT_Byte,
                )

                # Copy geotransform and projection
                output_dataset.SetGeoTransform(raster_layer.GetGeoTransform())
                output_dataset.SetProjection(raster_layer.GetProjection())

                drv = ogr.GetDriverByName("ESRI Shapefile")
                dst_ds = drv.CreateDataSource(os.path.join(self.in_dir, f"{f['start_time']}.shp"))

                band_2 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=0,
                    max=1,
                )
                band_2 = (band_2 - 0.0) / (1 - 0)

            elif f["channel"] == 3:
                band_3 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=0,
                    max=1,
                )
                band_3 = (band_3 - 0.0) / (1 - 0)

            elif f["channel"] == 4:
                band_4 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=0,
                    max=1,
                )
                band_4 = (band_4 - 0.0) / (1 - 0)

            elif f["channel"] == 5:
                band_5 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=0,
                    max=1,
                )
                band_5 = (band_5 - 0.0) / (1 - 0)

            elif f["channel"] == 6:
                band_6 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=0,
                    max=1,
                )
                band_6 = (band_6 - 0.0) / (1 - 0)

            elif f["channel"] == 7:
                mn, mm = 197.31, 411.86
                band_7 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_7 = (band_7 - mn) / (mm - mn)

            elif f["channel"] == 8:
                mn, mm = 138.05, 311.06
                band_8 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_8 = (band_8 - mn) / (mm - mn)

            elif f["channel"] == 9:
                mn, mm = 137.7, 311.08
                band_9 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_9 = (band_9 - mn) / (mm - mn)

            elif f["channel"] == 10:
                mn, mm = 126.91, 331.2
                band_10 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_10 = (band_10 - mn) / (mm - mn)

            elif f["channel"] == 11:
                mn, mm = 127.69, 341.3
                band_11 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_11 = (band_11 - mn) / (mm - mn)

            elif f["channel"] == 12:
                mn, mm = 117.49, 311.06
                band_12 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_12 = (band_12 - mn) / (mm - mn)

            elif f["channel"] == 13:
                mn, mm = 89.62, 341.28
                band_13 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_13 = (band_13 - mn) / (mm - mn)

            elif f["channel"] == 14:
                mn, mm = 96.19, 341.28
                band_14 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_14 = (band_14 - mn) / (mm - mn)

            elif f["channel"] == 15:
                mn, mm = 97.38, 341.28
                band_15 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
                    min=mn,
                    max=mm,
                )
                band_15 = (band_15 - mn) / (mm - mn)

            elif f["channel"] == 16:
                mn, mm = 92.7, 318.26
                band_16 = torch.clamp(
                    v2.ToImage()(Image.open(band)),
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

        return inputs, [output_dataset, dst_ds, f['start_time']]


class CustomDataset(Dataset):
    def __init__(
        self, datalist: List[ModelInput], transforms=None, target_transforms=None
    ) -> None:
        super().__init__()
        self.datalist = datalist
        self.transform = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        input_images, label = self.datalist[index].create_input()

        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            input_images = self.transform(input_images)

        return input_images, label, self.datalist[index].in_dir

def _split_file(path):
    dic = defaultdict(list)

    for file in os.listdir(path):
        if 'output' in file or '.png' in file or ':' in file:
            continue

        f = parse_filename(file.split("/")[-1])
        dic[f["start_time"]].append(file)

    return dic


if __name__ == "__main__":
    _model_path = os.getenv("MODEL_PATH")
    if _model_path is not None:
        model_path = _model_path
    else:
        raise ValueError("MODEL_PATH value not found")

    MODEL = torch.load(model_path, map_location=torch.device('cpu'))
    MODEL.eval()

    _threshold = os.getenv("THRESHOLD")
    if _threshold is not None:
        threshold = float(_threshold)
    else:
        raise ValueError("THRESHOLD value not found")

    _patch_dir = os.getenv("BASE_PATCHES_DIR")
    if _patch_dir is not None:
        patch_dir = _patch_dir
    else:
        raise ValueError("PATCH_DIR value not found")

    fires = []
    for fire in os.listdir(patch_dir):
        if len(os.listdir(os.path.join(patch_dir, fire))) > 16 and len(os.listdir(os.path.join(patch_dir, fire))) % 16 == 0:
            d = _split_file(os.path.join(patch_dir, fire))
            for k,v in d.items():
                fires.append(ModelInput([os.path.join(patch_dir, fire, x) for x in v]))
        else:
            fires.append(ModelInput(os.path.join(patch_dir, fire)))

        d = _split_file(os.path.join(patch_dir, fire))

        for k,v in d.items():

            dict = {}
            for file in v:
                f = parse_filename(file)
                dict[f["channel"]] = file

            final_image = Image.new(
                "RGB",
                (560, 32),
                color="black")

            final_image.paste(rgb_image(dict, os.path.join(patch_dir, fire)), (0, 0))

            for i in range(1, 17):
                file = dict[i]
                img_path = os.path.join(patch_dir, fire, file)
                f = parse_filename(file)
                img = Image.open(img_path)
                im = np.array(img).tolist()
                im = analysis_utils.scale_values(im, f["channel"])
                im = np.uint8(cm(im) * 255)
                im = Image.fromarray(im)
                final_image.paste(im, (i * 33, 0))

            final_image.save(f"{patch_dir}/{fire}/{str(k)}.png")

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    dataset = CustomDataset(fires, transforms=transform, target_transforms=transform)

    geojson_featurecollection = {
                                    'type': 'FeatureCollection',
                                    'features': []
                                }

    for i in tqdm(dataset):
        inputs, temp, dir = i
        gtiff, shp, t = temp

        with torch.no_grad():
            img = torch.unsqueeze(inputs, 0)
            img = transform(img)
            out = torch.sigmoid(MODEL(img))
            out[out < threshold] = 0.0
            out[out >= threshold] = 1.0
            out = torch.squeeze(out, 0)
            out = to_pil_image(out)
            im_label = np.array(out)

            gtiff.GetRasterBand(1).WriteArray(im_label)
            gtiff.FlushCache()

            data = gtiff.GetRasterBand(1)

            srs = osr.SpatialReference()
            srs.ImportFromWkt(gtiff.GetProjection())

            dst_layer = shp.CreateLayer("output", srs=srs)
            gdal.Polygonize(data, data, dst_layer, -1, [], callback=None)

            if np.sum(im_label) > 0:
                for feature in dst_layer:
                    print("Feature found")
                    feature_dict = feature.ExportToJson(as_object=True)
                    feature_dict['properties']['id'] = dir.split("/")[-1]
                    feature_dict['properties']['timestamp'] = t
                    geojson_featurecollection["features"].append(feature_dict)

            shp.Destroy()   

    # saving shp file as geojson file
    hotspot_base_dir = os.getenv("BASE_PERIMETERS_DIR")
    if not os.path.exists(hotspot_base_dir):
        os.mkdir(hotspot_base_dir)

    timestamp_str = os.getenv('TIMESTAMP')
    date_obj = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    gj_path = f'{os.getenv("BASE_PERIMETERS_DIR")}/hotspots_{str(date_obj)}.json'
    with open(gj_path, 'w') as f:
        json.dump(geojson_featurecollection, f, default=str)