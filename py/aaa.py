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
from py.restore_signal import restore_cs1_signal, restore_cs1_signal_gurobi, GurobiInFeasibleException, restore_cs1_signal_cosamp


def read_cache(*pairs, dry=False, restoration_type=None):
    if dry:
        return {(s, m): {} for s, m in pairs}

    assert isinstance(restoration_type, str)

    print(f"{datetime.utcnow().isoformat()} Started LP cache initialization")

    cache = {}

    for s, m in pairs:
        sub_cache = {}
        try:
            cache_path = Path(f"/home/rolandw0w/Development/PhD/cache/{s}_{m}-{restoration_type}")
            files = sorted(os.listdir(cache_path), key=lambda x: int(x.split("_")[1]))
            for chunk_file in files:
                with open(cache_path / chunk_file, "rb") as cache_file:
                    chunk = pickle.load(cache_file)
                    sub_cache.update(chunk)
        except FileNotFoundError:
            print(f"{datetime.utcnow().isoformat()} LP cache file not found")
            cache[(s, m)] = {}
        else:
            cache[(s, m)] = sub_cache

    return cache


# CACHE = read_cache()
# INIT_CACHE_SIZES = {k: len(v) for k, v in CACHE.items()}


def save_cache(all_cache, s, m, restoration_type):
    cache = all_cache[s, m]
    cache_path = Path(f"/home/rolandw0w/Development/PhD/cache/{s}_{m}-{restoration_type}")
    cache_path.mkdir(exist_ok=True)
    # cache = CACHE[(s, m)]
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


def restore_cs(mask_length: int, arrays: int, s: int, sdm: str, N: int, coef: int, restoration_type: str = "linprog", cache=None, max_arrays=None):
    print(f"{datetime.utcnow().isoformat()} Started K={mask_length} I={arrays} s={s} N={N} coef={coef} restoration_type={restoration_type}")
    m = coef*s
    matrix = dw.get_transformation(f"{SYNTH_ROOT}/{sdm}/s{s}/matrix_{m}.csv")
    matrix_transposed = np.transpose(matrix)

    restored = read_arrays(f"{SYNTH_ROOT}/{sdm}/s{s}/read_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter="->", type_=np.float32)
    clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv", type_=np.int16, max_arrays=max_arrays)
    indices = read_indices(f"{SYNTH_ROOT}/{sdm}/s{s}/read_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter="->")
    clean = [clean[x] for x in indices]
    restored_noisy = read_arrays(f"{SYNTH_ROOT}/{sdm}/s{s}/read_noisy_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                                 delimiter="->", type_=np.float32)

    clean_signals = []
    restored_signals = []
    restored_noisy_signals = []
    # matrix = np.vstack([matrix, np.ones((600,))])
    if cache is None:
        cache = read_cache((s, m), restoration_type=restoration_type)
    init_cache_sizes = {(s, m): len(v) for (s, m), v in cache.items()}
    print(f"{datetime.utcnow().isoformat()} Initial caches sizes:",
          json.dumps({f"{s}-{m}": v for (s, m), v in init_cache_sizes.items()}, indent=None)
          )
    cache_hits = 0
    cache_hits_noisy = 0

    lin_prog_time = 0
    lin_prog_time_noisy = 0
    clean_noisy = 0
    ci = []
    ni = []
    log_freq = 10_000
    checkpoint_freq = 10_000
    for i in range(arrays):
        if (i + 1) % log_freq == 0:
            print(f"\t{datetime.utcnow().isoformat()} In process {i + 1} K={mask_length} I={arrays} s={s} N={N} coef={coef}")
        clean_signal = np.zeros((600,), dtype=np.int8)
        for ind in clean[i]:
            clean_signal[ind] = 1

        rest = np.rint(restored[i]).astype(np.int8)
        rest_noisy = np.rint(restored_noisy[i]).astype(np.int8)

        key = rest.tobytes()
        key_noisy = rest_noisy.tobytes()

        cached = cache[(s, m)].get(key)
        if cached is None:
            t = time()

            if restoration_type == "cosamp":
                restored_signal, iterations = restore_cs1_signal_cosamp(clean_signal.nonzero(), rest, matrix, max_iter=5, transformation_transposed=matrix_transposed)
            elif restoration_type == "linprog":
                restored_signal = restore_cs1_signal(clean_signal.nonzero(), rest, matrix, restoration_type="linprog")
            else:
                raise Exception()
            # try:
            #     restored_signal = restore_cs1_signal_gurobi(clean_signal.nonzero(), rest, matrix)
            # except GurobiInFeasibleException:
            #     restored_signal = restore_cs1_signal(clean_signal.nonzero(), rest, matrix,
            #                                          restoration_type=restoration_type)
            #     cgi += 1
            cache[(s, m)][key] = restored_signal
            lin_prog_time += (time() - t)
            # ci.append(iterations)
        else:
            restored_signal = cached
            cache_hits += 1

        cached_noisy = cache[(s, m)].get(key_noisy)
        if cached_noisy is None:
            t = time()
            if restoration_type == "cosamp":
                restored_noisy_signal, iterations = restore_cs1_signal_cosamp(clean_signal.nonzero(), rest_noisy, matrix, max_iter=5, transformation_transposed=matrix_transposed)
            elif restoration_type == "linprog":
                restored_noisy_signal = restore_cs1_signal(clean_signal.nonzero(), rest_noisy, matrix, restoration_type="linprog")
            else:
                raise Exception()
            # try:
            #     restored_noisy_signal = restore_cs1_signal_gurobi(clean_signal.nonzero(), rest_noisy, matrix)
            # except GurobiInFeasibleException:
            #     restored_noisy_signal = restore_cs1_signal(clean_signal.nonzero(), rest_noisy, matrix,
            #                                                restoration_type=restoration_type)
            #     ngi += 1
            cache[(s, m)][key_noisy] = restored_noisy_signal
            lin_prog_time_noisy += (time() - t)
            # ni.append(iterations)
        else:
            restored_noisy_signal = cached_noisy
            cache_hits_noisy += 1

        clean_signals.append(clean_signal)
        restored_signals.append(restored_signal)
        restored_noisy_signals.append(restored_noisy_signal)

        if (i + 1) % checkpoint_freq == 0:
            temp_res = _check(clean_signals, restored_signals, restored_noisy_signals, i + 1, s, sdm, N, k=mask_length)
            temp_res["clean_iterations"] = np.mean(ci) if ci else 0
            temp_res["noisy_iterations"] = np.mean(ni) if ni else 0
            temp_res["clean_errors"] = np.count_nonzero(ci == -1)
            temp_res["noisy_errors"] = np.count_nonzero(ni == -1)
            if len(cache[(s, m)]) > init_cache_sizes.get((s, m), 0):
                save_cache(cache, s, m, restoration_type)
            print(f"{datetime.utcnow().isoformat()} Finished K={mask_length} I={i + 1}/{arrays} s={s} N={N} coef={coef} res={json.dumps(temp_res)}")

    t = time()
    res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays, s, sdm, N, k=mask_length)
    res["coefficient"] = coef
    res["restoration_type"] = restoration_type
    res["cache_hits"] = cache_hits
    res["cache_hits_noisy"] = cache_hits_noisy
    res["clean!=noisy"] = clean_noisy
    res["lin_prog_time"] = lin_prog_time
    res["lin_prog_time_noisy"] = lin_prog_time_noisy
    res["metrics_time"] = time() - t

    print(f"{datetime.utcnow().isoformat()} Finished K={mask_length} I={arrays} s={s} N={N} coef={coef}  res={json.dumps(res)}")
    return res


def restore_cs_nat(mask_length: int, arrays: int, m: int, sdm: str, N: int,
                   clean_signals=None, restoration_type: str = "linprog", method: str = None, cache=None):
    print(f"{datetime.utcnow().isoformat()} Started K={mask_length} I={arrays} N={N} m={m}")

    matrix = dw.get_transformation(f"{SYNTH_ROOT}/{sdm}/matrix_{m}.csv")

    restored = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter=None, type_=float)

    restored_noisy = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_noisy_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                                 delimiter=None, type_=float)

    restored_signals = []
    restored_noisy_signals = []

    matrix_transposed = matrix.transpose()

    for i in range(arrays):
        clean_signal = clean_signals[i]

        num_ones = len(clean_signal.nonzero()[0])

        # rest = np.rint(restored[i]).astype(int)
        # rest_noisy = np.rint(restored_noisy[i]).astype(int)

        # if restoration_type == "linprog":
        #     rest = dw.round_to_odd_or_even_arr(restored[i], num_ones)
        #     rest_noisy = dw.round_to_odd_or_even_arr(restored_noisy[i], num_ones)
        # else:

        rest = np.rint(restored[i]).astype(np.int8)
        rest_noisy = np.rint(restored_noisy[i]).astype(np.int8)

        if restoration_type == "cosamp":
            restored_signal, _ = restore_cs1_signal_cosamp(clean_signal.nonzero(), rest, matrix, max_iter=5,
                                                           transformation_transposed=matrix_transposed)
        elif restoration_type == "linprog":
            restored_signal = restore_cs1_signal(clean_signal.nonzero(), rest, matrix, restoration_type="linprog")

        if restoration_type == "cosamp":
            restored_noisy_signal, _ = restore_cs1_signal_cosamp(clean_signal.nonzero(), rest_noisy, matrix, max_iter=5,
                                                                 transformation_transposed=matrix_transposed)
        elif restoration_type == "linprog":
            restored_noisy_signal = restore_cs1_signal(clean_signal.nonzero(), rest_noisy, matrix, restoration_type="linprog")

        clean_signals.append(clean_signal)
        restored_signals.append(restored_signal)
        restored_noisy_signals.append(restored_noisy_signal)

    t = time()
    res = _check_no_cache(clean_signals, restored_signals, restored_noisy_signals, arrays, 0, sdm, N, k=mask_length)
    res["m"] = m
    res["restoration_type"] = restoration_type
    res["metrics_time"] = time() - t

    print(f"{datetime.utcnow().isoformat()} Finished K={mask_length} I={arrays} N={N} m={m}  res={json.dumps(res)}")
    return res


def restore_cs_mixed(mask_length: int, arrays: int, m: int, sdm: str, N: int, restoration_type: str = "linprog", cache=None, max_arrays=None):
    print(f"{datetime.utcnow().isoformat()} Started K={mask_length} I={arrays} N={N} M={m} restoration_type={restoration_type}")
    matrix = dw.get_transformation(f"{SYNTH_ROOT}/{sdm}/matrix_{m}.csv")
    matrix_transposed = np.transpose(matrix)

    restored = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter=None, type_=np.float32)
    clean_12 = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_12.csv", type_=np.int16, max_arrays=max_arrays)
    clean_16 = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_16.csv", type_=np.int16, max_arrays=max_arrays)
    clean_20 = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_20.csv", type_=np.int16, max_arrays=max_arrays)
    clean = []
    for i in range(arrays // 3):
        clean.append(clean_12[i])
        clean.append(clean_16[i])
        clean.append(clean_20[i])

    restored_noisy = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_noisy_m_{m}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                                 delimiter=None, type_=np.float32)

    clean_signals = []
    restored_signals = []
    restored_noisy_signals = []
    # matrix = np.vstack([matrix, np.ones((600,))])
    cache_key = ("mixed", m)
    if cache is None:
        cache = read_cache(cache_key, restoration_type=restoration_type)
    init_cache_sizes = {(s, m): len(v) for (s, m), v in cache.items()}
    print(f"{datetime.utcnow().isoformat()} Initial caches sizes:",
          json.dumps({f"{s}-{m}": v for (s, m), v in init_cache_sizes.items()}, indent=None)
          )
    cache_hits = 0
    cache_hits_noisy = 0

    lin_prog_time = 0
    lin_prog_time_noisy = 0
    clean_noisy = 0
    ci = []
    ni = []
    log_freq = 10_000
    checkpoint_freq = 10_000
    for i in range(arrays):
        if (i + 1) % log_freq == 0:
            print(f"\t{datetime.utcnow().isoformat()} In process {i + 1} K={mask_length} I={arrays} N={N} M={m}")
        clean_signal = np.zeros((600,), dtype=np.int8)
        for ind in clean[i]:
            clean_signal[ind] = 1

        rest = np.rint(restored[i]).astype(np.int8)
        rest_noisy = np.rint(restored_noisy[i]).astype(np.int8)

        key = rest.tobytes()
        key_noisy = rest_noisy.tobytes()

        cached = cache[cache_key].get(key)
        if cached is None:
            t = time()

            if restoration_type == "cosamp":
                restored_signal, iterations = restore_cs1_signal_cosamp(clean_signal.nonzero(), rest, matrix, max_iter=5, transformation_transposed=matrix_transposed)
            elif restoration_type == "linprog":
                restored_signal = restore_cs1_signal(clean_signal.nonzero(), rest, matrix, restoration_type="linprog")
            else:
                raise Exception()
            # try:
            #     restored_signal = restore_cs1_signal_gurobi(clean_signal.nonzero(), rest, matrix)
            # except GurobiInFeasibleException:
            #     restored_signal = restore_cs1_signal(clean_signal.nonzero(), rest, matrix,
            #                                          restoration_type=restoration_type)
            #     cgi += 1
            cache[cache_key][key] = restored_signal
            lin_prog_time += (time() - t)
            # ci.append(iterations)
        else:
            restored_signal = cached
            cache_hits += 1

        cached_noisy = cache[cache_key].get(key_noisy)
        if cached_noisy is None:
            t = time()
            if restoration_type == "cosamp":
                restored_noisy_signal, iterations = restore_cs1_signal_cosamp(clean_signal.nonzero(), rest_noisy, matrix, max_iter=5, transformation_transposed=matrix_transposed)
            elif restoration_type == "linprog":
                restored_noisy_signal = restore_cs1_signal(clean_signal.nonzero(), rest_noisy, matrix, restoration_type="linprog")
            else:
                raise Exception()
            # try:
            #     restored_noisy_signal = restore_cs1_signal_gurobi(clean_signal.nonzero(), rest_noisy, matrix)
            # except GurobiInFeasibleException:
            #     restored_noisy_signal = restore_cs1_signal(clean_signal.nonzero(), rest_noisy, matrix,
            #                                                restoration_type=restoration_type)
            #     ngi += 1
            cache[cache_key][key_noisy] = restored_noisy_signal
            lin_prog_time_noisy += (time() - t)
            # ni.append(iterations)
        else:
            restored_noisy_signal = cached_noisy
            cache_hits_noisy += 1

        clean_signals.append(clean_signal)
        restored_signals.append(restored_signal)
        restored_noisy_signals.append(restored_noisy_signal)

        if (i + 1) % checkpoint_freq == 0:
            temp_res = _check(clean_signals, restored_signals, restored_noisy_signals, i + 1, m, sdm, N, k=mask_length)
            temp_res["clean_iterations"] = np.mean(ci) if ci else 0
            temp_res["noisy_iterations"] = np.mean(ni) if ni else 0
            temp_res["clean_errors"] = np.count_nonzero(ci == -1)
            temp_res["noisy_errors"] = np.count_nonzero(ni == -1)
            if len(cache[cache_key]) > init_cache_sizes.get(cache_key, 0):
                save_cache(cache, "mixed", m, restoration_type)
            print(f"{datetime.utcnow().isoformat()} Finished K={mask_length} I={i + 1}/{arrays} N={N} M={m} res={json.dumps(temp_res)}")

    t = time()
    res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays, m, sdm, N, k=mask_length)
    res["m"] = m
    res["restoration_type"] = restoration_type
    res["cache_hits"] = cache_hits
    res["cache_hits_noisy"] = cache_hits_noisy
    res["clean!=noisy"] = clean_noisy
    res["lin_prog_time"] = lin_prog_time
    res["lin_prog_time_noisy"] = lin_prog_time_noisy
    res["metrics_time"] = time() - t

    print(f"{datetime.utcnow().isoformat()} Finished K={mask_length} I={arrays} N={N} M={m}  res={json.dumps(res)}")
    return res


def read_sdm(mask_length: int, arrays: int, s: int, sdm: str, N: int, clean_signals = None):
    print(f"{datetime.utcnow().isoformat()} Started K={mask_length} I={arrays} s={s}")

    restored = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_S_{s}_K_{mask_length}_I_{arrays}.csv", delimiter="->")
    if clean_signals is None:
        clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv")
        indices = read_indices(f"{SYNTH_ROOT}/{sdm}/read_S_{s}_K_{mask_length}_I_{arrays}.csv", delimiter="->")
        clean = [clean[x] for x in indices]
    restored_noisy = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_noisy_S_{s}_K_{mask_length}_I_{arrays}.csv", delimiter="->")

    if isinstance(clean_signals, (list, np.ndarray)):
        calc = False
    else:
        clean_signals = []
        calc = True

    restored_signals = []
    restored_noisy_signals = []
    for i in range(arrays):
        if calc:
            clean_signal = np.zeros((600,))
            for ind in clean[i]:
                clean_signal[ind] = 1
            clean_signals.append(clean_signal)

        restored_signals.append(np.array(restored[i]))
        restored_noisy_signals.append(np.array(restored_noisy[i]))

    # with open(f"{SYNTH_ROOT}/{sdm}/signals/signal_S_{s}_m_{m}_K_{mask_length}_I_{arrays}.csv", "w") as f:
    #     for restored_signal in restored_signals:
    #         line = ",".join([str(int(x)) for x in restored_signal])
    #         f.write(line + "\n")
    #
    # with open(f"{SYNTH_ROOT}/{sdm}/signals_noisy/signal_noisy_S_{s}_m_{m}_K_{mask_length}_I_{arrays}.csv", "w") as f_noisy:
    #     for restored_noisy_signal in restored_noisy_signals:
    #         line = ",".join([str(int(x)) for x in restored_noisy_signal])
    #         f_noisy.write(line + "\n")

    res = _check_no_cache(clean_signals, restored_signals, restored_noisy_signals, arrays, s, sdm, N, k=mask_length)

    print(f"{datetime.utcnow().isoformat()} Finished K={mask_length} I={arrays} s={s} N={N} res={json.dumps(res)}")
    return res


def run_cs():
    l = [
        [12, [[8, 50], [12, 40], [6, 60]]],
        [16, [[8, 40], [12, 30], [6, 50]]],
        [20, [[8, 30], [12, 20], [6, 40]]],

        # [5, [[16, 45], [14, 51], [12, 57], [10, 63], [8, 69]]],
        # [6, [[16, 37], [14, 43], [12, 49], [10, 55], [8, 61]]],
        # [7, [[16, 32], [14, 36], [12, 40], [10, 44], [8, 48]]],
        # [8, [[16, 28], [14, 32], [12, 36], [10, 40], [8, 44]]],
        # [9, [[16, 25], [14, 27], [12, 29], [10, 31], [8, 33]]],
        # [10,[[16, 23], [14, 25], [12, 27], [10, 29], [8, 31]]],
    ]

    # a = [
    #     restore_cs(8, 500, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     restore_cs(8, 1000, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     restore_cs(8, 1500, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     restore_cs(8, 2000, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     restore_cs(8, 2500, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     restore_cs(8, 3000, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     # restore_cs(6, 3500, 10, "cs_conf3", 23 * 1_000_000, 16),
    #     # restore_cs(6, 4000, 10, "cs_conf3", 23 * 1_000_000, 16),
    # ]
    # print(a)
    records = []
    rest_type = "cosamp"
    mask_lengths = [150, 200, 250, 300, 175, 225, 275, 325, 350]
    arrays = [2*1000, 4*1000, 6*1000, 8*1000, 10*1000]
    for K in mask_lengths:
        for num_ones, items in l:
            for coef, n in items:
                m = coef*num_ones
                cache = read_cache((num_ones, m), restoration_type=rest_type)
                coef_n_list = []
                for arrays_num in arrays:  # range(10_000, 200_000 + 1, 10_000): 2100*1000, 2200*1000, 2300*1000, 2400*1000,
                    try:
                        record = restore_cs(K, arrays_num, num_ones, "cs_conf_reverse", n * 1_000_000, coef,
                                            restoration_type=rest_type, cache=cache, max_arrays=max(arrays))
                    except FileNotFoundError as fnfe:
                        print("FileNotFoundError", fnfe)
                        continue
                    records.append(record)
                    coef_n_list.append(record)
                if not coef_n_list:
                    print("Empty", K, num_ones, coef, n)
                    continue
                coef_n_df = pd.DataFrame(coef_n_list)
                coef_n_df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse/s_{num_ones}_K_{K}_coef_{coef}_{rest_type}_small.csv",
                                 index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["features_count", "coefficient", "arrays_count"])
    df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse/temp_{rest_type}_small.csv", index=False)


def run_cs_mixed():
    l = [
        ("cosamp", 150, 40),
        ("cosamp", 120, 60),
        ("cosamp", 100, 80),
        ("cosamp", 75, 100),
        ("linprog", 100, 80),
        ("linprog", 75, 100),
    ]

    records = []
    K = 4
    mask_lengths = [K]
    arrays = [30*1000, 60*1000, 90*1000, 120*1000, 150*1000]
    for rest_type, m, n in l:
        cache = read_cache(("mixed", m), restoration_type=rest_type)
        coef_n_list = []
        for arrays_num in arrays:  # range(10_000, 200_000 + 1, 10_000): 2100*1000, 2200*1000, 2300*1000, 2400*1000,
            try:
                record = restore_cs_mixed(K, arrays_num, m, "cs_conf4_mixed", n * 1_000_000, restoration_type=rest_type, cache=cache, max_arrays=max(arrays))
            except FileNotFoundError as fnfe:
                print("FileNotFoundError", fnfe)
                continue
            records.append(record)
            coef_n_list.append(record)
        if not coef_n_list:
            print("Empty", K, m, n)
            continue
        coef_n_df = pd.DataFrame(coef_n_list)
        coef_n_df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/m_{m}_K_{K}_{rest_type}.csv",
                         index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["m", "restoration_type", "arrays_count"])
    df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/cs_mixed.csv", index=False)


def run_jaeckel():
    params = []
    for s in [10, 4, 5, 6, 7, 8, 9]:
        for i in range(500, 25_001, 500):
            params.append((3, i, s, "jaeckel", 8_000_000))

    # read_sdm(*params[0])
    with mp.Pool(32) as pool:
        res = pool.starmap(read_sdm, params)
    df = pd.DataFrame(res).sort_values(by=["features_count", "arrays_count"])
    df = df.sort_values(by=["features_count", "arrays_count"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/jaeckel.csv", index=False)


def run_kanerva():
    params = []
    for s in [4, 5, 6, 7, 8, 9, 10]:
        for i in range(500, 25_001, 500):
            params.append((7, i, s, "kanerva", 8_000_000))

    # read_sdm(*params[0])
    with mp.Pool(32) as pool:
        res = pool.starmap(read_sdm, params)
    df = pd.DataFrame(res).sort_values(by=["features_count", "arrays_count"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/kanerva.csv", index=False)


def run_jaeckel_nat():
    clean = dw.get_features("../data/features.bin")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break
    res = []
    for i in range(250, 5_001, 250):
        r = read_sdm(3, i, 4, "jaeckel_nat", 8_000_000, clean_list)
        res.append(r)
        # params.append((7, i, 4, "kanerva_nat", 8_000_000, clean))

    # read_sdm(*params[0])
    # with mp.Pool(32) as pool:
    #     res = pool.starmap(read_sdm, params)
    df = pd.DataFrame(res).sort_values(by=["arrays_count"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/jaeckel_nat.csv", index=False)


def run_kanerva_nat():
    clean = dw.get_features("../data/features.bin")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break
    res = []
    for i in range(250, 5_001, 250):
        r = read_sdm(7, i, 4, "kanerva_nat", 8_000_000, clean_list)
        res.append(r)
        # params.append((7, i, 4, "kanerva_nat", 8_000_000, clean))

    # read_sdm(*params[0])
    # with mp.Pool(32) as pool:
    #     res = pool.starmap(read_sdm, params)
    df = pd.DataFrame(res).sort_values(by=["arrays_count"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/kanerva_nat.csv", index=False)


def run_cs3_nat():
    clean = dw.get_features("../data/features.bin")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break

    l = [(100, 36)]
    mask_lengths = [8, 10, 12, 14, 16]
    records = []
    for mask_length in mask_lengths:
        for m, n in l[::-1]:
            params = []
            for arrays_num in range(250, 5_001, 250):
                params.append((mask_length, arrays_num, m, "cs3_nat", n * 1_000_000, clean_list, "omp"))
            for arrays_num in range(250, 5_001, 250):
                params.append((mask_length, arrays_num, m, "cs3_nat", n * 1_000_000, clean_list, "linprog", "highs-ds"))
                # record = restore_cs_nat(3, arrays_num, m, "cs_nat", n * 1_000_000, clean_signals=clean_list)
            # restore_cs_nat(*params[0])
            omp = params[:len(params)//2]
            linprog = params[len(params)//2:]
            with mp.Pool(32) as pool:
                coef_n_list = pool.starmap(restore_cs_nat, omp)
                records += coef_n_list
            with mp.Pool(1) as pool:
                coef_n_list = pool.starmap(restore_cs_nat, linprog)
                records += coef_n_list

            coef_n_df = pd.DataFrame(coef_n_list)
            coef_n_df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs3_nat/m_{m}_N_{n}.csv",
                             index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["m", "arrays_count", "restoration_type"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/cs3_nat.csv", index=False)


def run_cs_nat():
    clean = dw.get_features("../data/features.bin")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break

    l = [(150, 25), (100, 40), (75, 45), (60, 50)]
    records = []
    for m, n in l:
        params = []
        for arrays_num in range(250, 5_001, 250):
            params.append((3, arrays_num, m, "cs_nat", n * 1_000_000, clean_list, "omp"))
        for arrays_num in range(250, 5_001, 250):
            params.append((3, arrays_num, m, "cs_nat", n * 1_000_000, clean_list, "linprog", "highs-ds"))
            # record = restore_cs_nat(3, arrays_num, m, "cs_nat", n * 1_000_000, clean_signals=clean_list)
        # restore_cs_nat(*params[0])
        omp = params[:len(params)//2]
        linprog = params[len(params)//2:]
        # with mp.Pool(10) as pool:
        #     coef_n_list = pool.starmap(restore_cs_nat, omp)
        #     records += coef_n_list
        with mp.Pool(1) as pool:
            coef_n_list = pool.starmap(restore_cs_nat, linprog)
            records += coef_n_list

        coef_n_df = pd.DataFrame(coef_n_list)
        coef_n_df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_nat/m_{m}_N_{n}.csv",
                         index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["m", "arrays_count", "restoration_type"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/cs_nat.csv", index=False)


def run_cs_nat_alpha():
    c = dw.get_features("../data/features.bin")
    clean = dw.get_features_from_txt("../data/features.txt")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break

    l = [(150, 25), (100, 35), (75, 45), (60, 50)]
    records = []
    for m, n in l[::-1]:
        params = []
        for arrays_num in range(250, 5_000, 250):
            params.append((3, arrays_num, m, "cs_nat_alpha", n * 1_000_000, clean_list, "omp"))
        for arrays_num in range(250, 5_000, 250):
            params.append((3, arrays_num, m, "cs_nat_alpha", n * 1_000_000, clean_list, "linprog", "highs-ds"))
            # record = restore_cs_nat(3, arrays_num, m, "cs_nat", n * 1_000_000, clean_signals=clean_list)
        # restore_cs_nat(*params[0])
        omp = params[:len(params)//2]
        linprog = params[len(params)//2:]
        # with mp.Pool(1) as pool:
        #     coef_n_list = pool.starmap(restore_cs_nat, omp)
        #     records += coef_n_list
        with mp.Pool(1) as pool:
            coef_n_list = pool.starmap(restore_cs_nat, linprog)
            records += coef_n_list

        coef_n_df = pd.DataFrame(coef_n_list)
        coef_n_df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_nat_alpha/m_{m}_N_{n}.csv",
                         index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["m", "arrays_count", "restoration_type"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/cs_nat_alpha.csv", index=False)


def run_cs_nat_bi():
    clean = dw.get_features("../data/features.bin")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break

    l = [(150, 28),  (100, 36), (75, 45), (60, 52)]
    records = []
    for m, n in l[::-1]:
        params = []
        for arrays_num in range(250, 5_001, 250):
            params.append((3, arrays_num, m, "cs_nat_balanced_impact", n * 1_000_000, clean_list, "omp"))
        for arrays_num in range(250, 5_001, 250):
            params.append((3, arrays_num, m, "cs_nat_balanced_impact", n * 1_000_000, clean_list, "linprog", "highs-ds"))
            # record = restore_cs_nat(3, arrays_num, m, "cs_nat", n * 1_000_000, clean_signals=clean_list)
        # restore_cs_nat(*params[0])
        omp = params[:len(params)//2]
        linprog = params[len(params)//2:]
        with mp.Pool(32) as pool:
            coef_n_list = pool.starmap(restore_cs_nat, omp)
            records += coef_n_list
        with mp.Pool(1) as pool:
            coef_n_list = pool.starmap(restore_cs_nat, linprog)
            records += coef_n_list

        coef_n_df = pd.DataFrame(coef_n_list)
        coef_n_df.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_nat_balanced_impact/m_{m}_N_{n}.csv",
                         index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["m", "arrays_count", "restoration_type"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/cs_nat_balanced_impact.csv", index=False)


def run_cs_reverse_nat():
    clean = dw.get_features("../data/features.bin")
    clean_list = []
    for i in range(clean.shape[1]):
        arr = clean[:, i]
        if sum(arr) < 4:
            continue
        clean_list.append(arr)
        if len(clean_list) == 5_000:
            break

    mask_lengths = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    arrays_nums = [1000, 2000, 3000, 4000, 5000]
    pairs = [(150, 40), (100, 50), (75, 60)]

    records_list = []
    for mask_length in reversed(mask_lengths):
        records_mask = []
        for M, N in pairs:
            for arrays_num in arrays_nums:
                records = restore_cs_nat(mask_length, arrays_num, M, "cs_conf_reverse_nat", N * 1_000_000, clean_list, "cosamp")
                records_list.append(records)
                records_mask.append(records)

        df_mask = pd.DataFrame(records_mask)
        df_mask = df_mask.sort_values(by=["mask_length", "m", "arrays_count", "restoration_type"])
        df_mask.to_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse_nat/aggr/records_{mask_length}.csv", index=False)

    df = pd.DataFrame(records_list)
    df = df.sort_values(by=["mask_length", "m", "arrays_count", "restoration_type"])
    df.to_csv("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse_nat/aggr/records.csv", index=False)


# run_jaeckel()
# run_jaeckel_nat()
# run_kanerva_nat()
# run_cs_nat()
# run_cs_nat_bi()
# run_cs3_nat()
# run_cs_nat_alpha()

#run_cs()

# run_cs_reverse_nat()
run_cs_mixed()
