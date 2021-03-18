import numpy as np


def calculate_l1(x: np.ndarray, y: np.ndarray):
    return np.abs(x-y).sum()
