import os
from typing import List

from download import Downloader, read_json_file
from fire import Fire

SMALL_FIRES_ID = [656, 19997, 36655, 50280, 16340]
MEDIUM_FIRES_ID = [25238, 18871, 4898, 2814, 16]
LARGE_FIRES_ID = [3183, 7881, 19827, 32882, 47511]
VERY_LARGE_FIRES_ID = [45226, 44040, 39178, 32186, 4602]


class Analytics_Downloader(Downloader):
    def __init__(self, fires: List[Fire], save_dir: str, fire_type:str, params: List[str]) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.fire_type_path = os.path.join(save_dir, fire_type)

        super().__init__(fires, self.fire_type_path, params)

        self.download(True)


if __name__ == "__main__":
    fires = read_json_file("./files/Filtered_WFIGS_Interagency_Perimeters.json")
    very_large_fires: List[Fire] = []
    large_fires: List[Fire] = []
    medium_fires: List[Fire] = []
    small_fires: List[Fire] = []

    for fire in fires:
        if fire.id in VERY_LARGE_FIRES_ID:
            very_large_fires.append(fire)
        if fire.id in LARGE_FIRES_ID:
            large_fires.append(fire)
        if fire.id in MEDIUM_FIRES_ID:
            medium_fires.append(fire)
        if fire.id in SMALL_FIRES_ID:
            small_fires.append(fire)

    Analytics_Downloader(very_large_fires, "analytics", "Very Large", ["ABI-L1b-RadC"])
    Analytics_Downloader(large_fires, "analytics", "Large", ["ABI-L1b-RadC"])
    Analytics_Downloader(medium_fires, "analytics", "Medium", ["ABI-L1b-RadC"])
    Analytics_Downloader(small_fires, "analytics", "Small", ["ABI-L1b-RadC"])
