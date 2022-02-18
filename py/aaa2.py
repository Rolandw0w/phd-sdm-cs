from datetime import datetime

from pathlib import Path

import numpy as np
import pandas as pd

from py.check import _check

SYNTH_ROOT = Path("/home/rolandw0w/Development/PhD/output/synth")
s_s = [12, 16, 20]


def read_arrays(path: str, delimiter: str = None, type_=int) -> object:
    arrays = []
    with open(path, "r") as f:
        content = f.read()
        lines = content.split("\n")

        for index, line in enumerate(lines[:-1]):
            if delimiter:
                line = line.split(delimiter)[1]
            try:
                array = np.array([type_(x) for x in line.split(",")])
            except:
                raise
            arrays.append(array)
    return arrays


def g(path, dtype=np.int8):
    with open(path, "r") as f:
        content = f.read()
        strings = content.split("\n")
        res = []
        for string in strings:
            if len(string) != 600:
                continue
            arr = [dtype(x) for x in string]
            np_arr = np.array(arr, dtype=dtype)
            res.append(np_arr)
        return res


def get_kanerva(N):
    kanerva_root = SYNTH_ROOT / "kanerva"
    try:
        df = pd.read_csv(kanerva_root / "kanerva.csv")
    except FileNotFoundError:
        records = []
        for s in s_s:
            clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv", type_=np.int16)
            clean_signals = []
            for arrays_count in list(range(250_000, 2_000_000 + 1, 250_000)):
                print(datetime.utcnow(), f"Started Kanerva (arrays_count={arrays_count})")

                restored_signals = g(kanerva_root / f"s{s}/read_R_{s-3}_I_{arrays_count}.csv", dtype=np.int8)
                # restored_signals = np.genfromtxt(kanerva_root / f"s{s}/read_K_4_I_{arrays_count}.csv", dtype=np.int8, delimiter="//")
                restored_noisy_signals = g(kanerva_root / f"s{s}/read_noisy_R_{s-3}_I_{arrays_count}.csv", dtype=np.int8)

                for i in range(arrays_count):
                    clean_signal = np.zeros((600,), dtype=np.int8)
                    for ind in clean[i]:
                        clean_signal[ind] = 1
                    clean_signals.append(clean_signal)
                res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays_count, s, "kanerva", N, k=4)
                records.append(res)
        df = pd.DataFrame(records)
        df.to_csv(kanerva_root / "kanerva.csv", index=False)

    return df


get_kanerva(15_000_000)
