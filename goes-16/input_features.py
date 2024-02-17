import numpy as np
from PIL import Image
from datetime import datetime
import os

class TiffFile:
    def __init__(self) -> None:
        self.band: int
        self.date: datetime
        self.file_name:str

    def __str__(self) -> str:
        return f"{self.band, self.date, self.file_name}"

    @classmethod
    def parse(cls, name:str):
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
        date_time = datetime(year=int(year),
                             month=int(month),
                             day=int(day),
                             hour=int(hour),
                             minute=int(minute),
                             second=int(seconds),
                             microsecond=int(microseconds))
        cls.date = date_time
        return cls

class InputFeatures:
    def __init__(self, save_dir) -> None:
        self.layers=[[3],
                     [3, 12],
                     [3, 13],
                     [3, 14],
                     [3, 15]]

        self.save_dir = save_dir
        self.days = [int(i) for i in os.listdir(f"{save_dir}") if i != "tmp"]
        self.features = {}
        for box in self.days:
            self.features[box] = os.listdir(f"{save_dir}/{box}/")[0]
        print(self.features)

    def get_file(self, file_list: list[str], band_id):
        for file in file_list:
           f = TiffFile.parse(file)
           if f.band == band_id:
               return file

    def get_features(self):
        for box, feature in self.features.items():
            time_file_dict = {}
            for file in os.listdir(f"{self.save_dir}/{box}/{feature}"):
               f = TiffFile.parse(file)
               time_file_dict[f.date] = []

            for file in os.listdir(f"{self.save_dir}/{box}/{feature}"):
                f = TiffFile.parse(file)
                time_file_dict[f.date].append(f.file_name)

            for date, files in time_file_dict.items():
                os.mkdir(f"{self.save_dir}/{box}/{str(date)}")
                for i, layer in enumerate(self.layers):
                    base_dir = f"{self.save_dir}/{box}/{feature}/"
                    if len(layer) == 1:
                        f = self.get_file(files, 3)
                        im = Image.open(os.path.join(base_dir, f))
                        im = im.resize((128, 128))
                        imarray = np.array(im)
                        im = Image.fromarray(imarray)
                        im.save(f"{self.save_dir}/{box}/{str(date)}/b_{i}.tiff")

                    else:
                        f1 = self.get_file(files, layer[0])
                        im1 = Image.open(os.path.join(base_dir, f1))
                        im1 = im1.resize((128, 128))
                        imarray1 = np.array(im1)


                        f2 = self.get_file(files, layer[1])
                        im2 = Image.open(os.path.join(base_dir, f2))
                        im2 = im2.resize((128, 128))
                        imarray2 = np.array(im2)

                        diff = imarray1 - imarray2
                        im = Image.fromarray(diff)
                        im.save(f"{self.save_dir}/{box}/{str(date)}/b_{i}.tiff")


if __name__ == "__main__":
    ip = InputFeatures("./DATA")
    ip.get_features()
