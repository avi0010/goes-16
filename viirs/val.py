from datetime import datetime, timedelta
import os

from pandas import date_range
from sklearn.cluster import DBSCAN

from viirs import ViirsDataset

class Val_ViirsDataset(ViirsDataset):
    def __init__(self, dir_location:str) -> None:
        super().__init__(dir_location, min_samples=8, eps=0.015)
        self.unique_dates = datetime(2024, 11, 8)

    def __clean_up(self, dir:str):
        if not os.path.exists(dir):
            return

        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    def download(self, base_dir:str, date_range:int=5, param:str="ABI-L1b-RadC", process=True):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.download_dir = os.path.join(self.base_dir, "downloads")
        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)

        for hr in range(-24, 49):
            date = self.unique_dates + timedelta(hours=hr)
            self.fit(date)
            self.download_datetime(date, self.download_dir, date_range, param, process)
            self.process_output(self.download_dir)
            self.patch(date)

    def fit(self, date: datetime):
        self.filtered_data_points = self.data_points
        tmp_data = [[x.lat, x.lon] for x in self.filtered_data_points]
        self._db = DBSCAN(eps=self.eps, min_samples=self.min_samples, algorithm="brute").fit(tmp_data)

    def download_datetime(self, date:datetime, save_dir:str, date_range:int = 1, param:str="ABI-L1b-RadC", process=True):
        days_since_year_start = (datetime(date.year, date.month, date.day) - datetime(date.year, 1, 1)).days + 1
            
        try:
            data_hour = self.fs.ls(
                f"s3://noaa-goes16/{param}/{date.year}/{str(days_since_year_start).zfill(3)}/{str(date.hour).zfill(2)}/"
            )

            dates = set(
                map(
                    lambda x: self.parse_filename(x.split("/")[-1])["start_time"],
                    data_hour,
                )
            )

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            nearest_time = self.nearest_min(date, dates)

            self.__clean_up(self.download_dir)

            for diff in nearest_time[:1]:
                closest_date = date + diff
                files = list(
                    filter(
                        lambda x: f"s{closest_date.strftime('%Y%j%H%M%S')}" in x, data_hour
                    )
                )

                assert len(files) == 16
                self.fs.get(files, save_dir)

            if process:
                self.process_dir(save_dir)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    shape_file = "./files/viirs_data/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.shp"
    dataset = Val_ViirsDataset(shape_file)
    dataset.download("ttttt", date_range=0)
