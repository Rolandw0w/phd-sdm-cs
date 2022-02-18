import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import os
import pandas as pd
from time import time

import data_wrangling as dw
from py.check import read_arrays, read_indices, SYNTH_ROOT, _check, _check_no_cache
from py.restore_signal import restore_cs1_signal

S, M = 8, 128

def read_cache(dry=False):
    if dry:
        return {}

    print(f"{datetime.utcnow().isoformat()} Started LP cache initialization")

    cache = {}

    for s, m in [(S, M)]:
        sub_cache = {}
        try:
            cache_path = Path(f"/home/rolandw0w/Development/PhD/cache/{s}-{m}")
            files = sorted(os.listdir(cache_path), key=lambda x: int(x.split("_")[1]))
            for chunk_file in files:
                with open(cache_path / chunk_file, "rb") as cache_file:
                    chunk = pickle.load(cache_file)
                    sub_cache.update(chunk)
        except FileNotFoundError:
            print(f"{datetime.utcnow().isoformat()} LP cache file not found")
            return {}

        cache[(s, m)] = sub_cache

    return cache


CACHE = read_cache()


def save_cache(s, m, C=CACHE):
    cache_path = Path(f"/home/rolandw0w/Development/PhD/cache/{s}_{m}")
    cache_path.mkdir(exist_ok=True)
    cache = C[(s, m)]
    cache_len = len(cache)
    chunk_size = 50_000
    n = (cache_len // chunk_size) + (0 if cache_len % chunk_size == 0 else 1)
    keys = list(cache.keys())

    print(f"{datetime.utcnow().isoformat()} Saving cache", end=": ")
    for i in range(n):
        left = i*chunk_size
        right = (i+1)*chunk_size
        keys_slice = keys[left:right]
        d = {key: cache[key] for key in keys_slice}

        with open(cache_path / f"chunk_{i}", "wb") as cache_file:
            pickle.dump(d, cache_file)
        print(i, end=" ")
    print()


d = {(S, M): {}}
for k, v in CACHE[(S, M)].items():
    arr = np.frombuffer(k, dtype=np.int64).astype(np.int8)
    key = arr.tobytes()
    vv = v.astype(np.int8)
    d[(S, M)][key] = vv
save_cache(S, M, d)
