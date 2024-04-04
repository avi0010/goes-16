from typing import List
import numpy as np


def scale_values(values: List[float], band:int):
    if band <= 6:
        mn = 0
        mm = 1.3

    elif band == 7:
        mn = 197.31
        mm = 411.86

    elif band == 8:
        mn = 138.05
        mm = 311.06

    elif band == 9:
        mn = 137.7
        mm = 311.08

    elif band == 10:
        mn = 126.91
        mm = 331.2

    elif band == 11:
        mn = 127.69
        mm = 341.3

    elif band == 12:
        mn = 117.49
        mm = 311.06

    elif band == 13:
        mn = 89.62
        mm = 341.27

    elif band == 14:
        mn = 96.19
        mm = 341.28

    elif band == 15:
        mn = 97.38
        mm = 341.28

    elif band == 16:
        mn = 92.7
        mm = 318.26

    else:
        raise ValueError(f"Band: {band} not a valid band")

    np_values = np.array(values)

    scaled_np_values = (np_values - mn) / (mm - mn)

    return scaled_np_values.tolist()
