from functools import partial

import math
import numpy as np


def get_transformation(matrix_path) -> np.ndarray:
    with open(matrix_path, "r") as f:
        content = f.read()
        transformation = [[int(x) for x in line.split(",")] for line in content.splitlines()]
        transformation_np = np.array(transformation)
        return transformation_np


def get_features(features_bin_path, left_slice: int = None) -> np.ndarray:
    with open(features_bin_path, "rb") as f_read:
        content = f_read.read()
        ords = ["{0:b}".format(b) for b in content]
        ords_normalized = [("0" * (8 - len(b)) + b) for b in ords]
        data_arr = [int(x) for x in "".join(ords_normalized)]
        l = []
        d = []
        c = 0
        for i, el in enumerate(data_arr):
            c += 1
            l.append(el)
            if c == 600:
                d.append(l)
                l = []
                c = 0

        data_np = np.column_stack(d)

        if left_slice:
            return data_np[:, :left_slice]
        return data_np


def get_restored(path) -> np.ndarray:
    with open(path, "r") as f:
        content = f.read()
        restored = [[float(x) for x in line.split(",")[:-1]] for line in content.splitlines()]
        restored_np = np.array(restored)
        return restored_np


def round_to_odd_or_even(num: float, num_ones: int):
    int_part = int(num)
    # frac_part = num - int_part

    if num_ones % 2 == 0:
        if int_part % 2 == 0:
            return int_part
        else:
            return int_part + int(np.sign(num))
    if num_ones % 2 == 1:
        if int_part % 2 == 0:
            return int_part + int(np.sign(num))
        else:
            return int_part

    #return round(num / 2) * 2 + (num_ones % 2)


def round_to_odd_or_even_arr(array: np.ndarray, num_ones: int):
    return np.vectorize(partial(round_to_odd_or_even, num_ones=num_ones))(array)
