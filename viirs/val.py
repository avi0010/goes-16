from datetime import datetime, timedelta
import os

from pandas import date_range

from viirs import ViirsDataset

class Val_ViirsDataset(ViirsDataset):
    def __init__(self, dir_location:str) -> None:
        super().__init__(dir_location, min_samples=5)

    def __clean_up(self, dir:str):
        if not os.path.exists(dir):
            return

        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    def download_datetime(self, date:datetime, save_dir:str, date_range:int = 1, param:str="ABI-L1b-RadC", process=True):
        for i in range(-1 * date_range, date_range+1):
            day = datetime(date.year, date.month, date.day) + timedelta(days=i)
            days_since_year_start = (datetime(day.year, day.month, day.day) - datetime(day.year, 1, 1)).days + 1
            
            try:
                data_hour = self.fs.ls(
                    f"s3://noaa-goes16/{param}/{date.year}/{str(days_since_year_start).zfill(3)}/00/"
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
    shape_file = "/home/aveekal/Downloads/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.shp"
    dataset = Val_ViirsDataset(shape_file)
    dataset.download("ttttt", date_range=0)
