import multiprocessing as mp

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import data_wrangling as dw
from py.restore_signal import restore_cs1_signal
from py.utils import perf_measure

SYNTH_ROOT = "/home/rolandw0w/Development/PhD/output/synth"


def read_arrays(path: str, delimiter: str = None, type_=int) -> object:
    arrays = []
    with open(path, "r") as f:
        content = f.read()
        lines = content.split("\n")

        for index, line in enumerate(lines[:-1]):
            if delimiter:
                line = line.split(delimiter)[1]
            try:
                array = [type_(x) for x in line.split(",")]
            except:
                raise
            arrays.append(array)
    return arrays


def read_indices(path: str, delimiter: str = None) -> list:
    inds = []
    with open(path, "r") as f:
        content = f.read()
        lines = content.split("\n")

        for index, line in enumerate(lines[:-1]):
            s = line.split(delimiter)[0]
            ind = int(s)
            inds.append(ind)
    return inds


CACHE = {}


def _check(clean, restored, restored_noisy, n, s, sdm, N, k):
    l1s = []
    fns = []
    fps = []
    exact = 0
    l1s_noisy = []
    fns_noisy = []
    fps_noisy = []
    exact_noisy = 0
    for i in range(n):
        clean_array = clean[i]
        restored_array = restored[i]
        restored_noisy_array = restored_noisy[i]

        key = (i, restored_array.tobytes())
        cached = CACHE.get(key)
        if cached is None:
            fn, fp, _, _ = perf_measure(clean_array, restored_array)
            CACHE[key] = (fn, fp)
        else:
            fn, fp = cached
        l1 = fn + fp

        key_noisy = (i, restored_noisy_array.tobytes())
        cached_noisy = CACHE.get(key_noisy)
        if cached_noisy is None:
            fn_noisy, fp_noisy, _, _ = perf_measure(clean_array, restored_noisy_array)
            CACHE[key_noisy] = (fn_noisy, fp_noisy)
        else:
            fn_noisy, fp_noisy = cached_noisy
        l1_noisy = fn_noisy + fp_noisy

        if l1 == 0:
            exact += 1
        if l1_noisy == 0:
            exact_noisy += 1

        l1s.append(l1)
        fns.append(fn)
        fps.append(fp)

        l1s_noisy.append(l1_noisy)
        fns_noisy.append(fn_noisy)
        fps_noisy.append(fp_noisy)

    round_num = 3
    avg_hammings = round(np.mean(l1s), round_num)
    avg_hammings_noisy = round(np.mean(l1s_noisy), round_num)

    avg_fp = round(np.mean(fps), round_num)
    avg_fp_noisy = round(np.mean(fps_noisy), round_num)

    avg_fn = round(np.mean(fns), round_num)
    avg_fn_noisy = round(np.mean(fns_noisy), round_num)

    e_p = round(100*exact/n, round_num)
    e_n_p = round(100*exact_noisy/n, round_num)

    # if k:
    #     msg = f"{sdm} | S={s} K={k} I={n} avg_l1={avg_hammings} avg_l1_n={avg_hammings_noisy} exact_%={e_p} exact_n_%={e_n_p}"
    # else:
    #     msg = f"{sdm} | S={s} I={n} avg_l1={avg_hammings} avg_l1_n={avg_hammings_noisy} exact_%={e_p} exact_n_%={e_n_p}"

    res = {
        "avg_fn": avg_fn,
        "avg_fn_noisy": avg_fn_noisy,
        "avg_fp": avg_fp,
        "avg_fp_noisy": avg_fp_noisy,
        "avg_l1": avg_hammings,
        "avg_l1_noisy": avg_hammings_noisy,
        "arrays_count": n,
        "cells_count": N,
        "exact": exact,
        "exact_noisy": exact_noisy,
        "exact_percent": e_p,
        "exact_noisy_percent": e_n_p,
        "features_count": s,
        "sdm_type": "jaeckel" if sdm == "labels" else sdm,
        "mask_length": k,
    }
    # print(msg)
    return res


def _check_no_cache(clean, restored, restored_noisy, n, s, sdm, N, k):
    l1s = []
    fns = []
    fps = []
    exact = 0
    l1s_noisy = []
    fns_noisy = []
    fps_noisy = []
    exact_noisy = 0
    for i in range(n):
        clean_array = clean[i]
        restored_array = restored[i]
        restored_noisy_array = restored_noisy[i]

        fn, fp, _, _ = perf_measure(clean_array, restored_array)
        l1 = fn + fp

        fn_noisy, fp_noisy, _, _ = perf_measure(clean_array, restored_noisy_array)
        l1_noisy = fn_noisy + fp_noisy

        if l1 == 0:
            exact += 1
        if l1_noisy == 0:
            exact_noisy += 1

        l1s.append(l1)
        fns.append(fn)
        fps.append(fp)

        l1s_noisy.append(l1_noisy)
        fns_noisy.append(fn_noisy)
        fps_noisy.append(fp_noisy)

    round_num = 3
    avg_hammings = round(np.mean(l1s), round_num)
    avg_hammings_noisy = round(np.mean(l1s_noisy), round_num)

    avg_fp = round(np.mean(fps), round_num)
    avg_fp_noisy = round(np.mean(fps_noisy), round_num)

    avg_fn = round(np.mean(fns), round_num)
    avg_fn_noisy = round(np.mean(fns_noisy), round_num)

    e_p = round(100*exact/n, round_num)
    e_n_p = round(100*exact_noisy/n, round_num)

    # if k:
    #     msg = f"{sdm} | S={s} K={k} I={n} avg_l1={avg_hammings} avg_l1_n={avg_hammings_noisy} exact_%={e_p} exact_n_%={e_n_p}"
    # else:
    #     msg = f"{sdm} | S={s} I={n} avg_l1={avg_hammings} avg_l1_n={avg_hammings_noisy} exact_%={e_p} exact_n_%={e_n_p}"

    res = {
        "avg_fn": avg_fn,
        "avg_fn_noisy": avg_fn_noisy,
        "avg_fp": avg_fp,
        "avg_fp_noisy": avg_fp_noisy,
        "avg_l1": avg_hammings,
        "avg_l1_noisy": avg_hammings_noisy,
        "arrays_count": n,
        "cells_count": N,
        "exact": exact,
        "exact_noisy": exact_noisy,
        "exact_percent": e_p,
        "exact_noisy_percent": e_n_p,
        "features_count": s,
        "sdm_type": "jaeckel" if sdm == "labels" else sdm,
        "mask_length": k,
    }
    # print(msg)
    return res


def check(mask_length: int, arrays: int, s: int, root: str, N: int):
    restored = read_arrays(f"{root}/read_S_{s}_K_{mask_length}_I_{arrays}.csv",
                           delimiter="->")
    clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv")
    indices = read_indices(f"{root}/read_S_{s}_K_{mask_length}_I_{arrays}.csv",
                           delimiter="->")
    clean = [clean[x] for x in indices]
    restored_noisy = read_arrays(f"{root}/read_noisy_S_{s}_K_{mask_length}_I_{arrays}.csv", delimiter="->")

    clean_arrays = []
    for clean_arr in clean:
        clean_array = np.zeros((600, ))
        for ind in clean_arr:
            clean_array[ind] = 1
        clean_arrays.append(clean_array)

    return _check(clean_arrays, restored, restored_noisy, arrays, s, N=N, sdm=root.split('/')[-1], k=mask_length)


def check_jaeckel(mask_length: int, arrays: int, s: int, n: int):
    return check(mask_length, arrays, s, f"{SYNTH_ROOT}/jaeckel", n)


def check_labels(mask_length: int, arrays: int, s: int, n: int):
    return check(mask_length, arrays, s, f"{SYNTH_ROOT}/labels", n)


def restore_cs(mask_length: int, arrays: int, s: int, sdm: str, N: int):
    matrix = dw.get_transformation(f"{SYNTH_ROOT}/{sdm}/matrix.csv")

    restored = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_S_{s}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter="->", type_=float)
    clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv")
    indices = read_indices(f"{SYNTH_ROOT}/{sdm}/read_S_{s}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter="->")
    clean = [clean[x] for x in indices]
    restored_noisy = read_arrays(f"{SYNTH_ROOT}/{sdm}/read_noisy_S_{s}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                                 delimiter="->", type_=float)

    clean_signals = []
    restored_signals = []
    restored_noisy_signals = []
    for i in range(arrays):
        clean_signal = np.zeros((600,))
        for ind in clean[i]:
            clean_signal[ind] = 1

        restored_signal = restore_cs1_signal(clean_signal.nonzero(), restored[i], matrix)
        restored_noisy_signal = restore_cs1_signal(clean_signal.nonzero(), restored_noisy[i], matrix)

        clean_signals.append(clean_signal)
        restored_signals.append(restored_signal)
        restored_noisy_signals.append(restored_noisy_signal)

    with open(f"{SYNTH_ROOT}/{sdm}/signals/signal_S_{s}_K_{mask_length}_I_{arrays}.csv", "w") as f:
        for restored_signal in restored_signals:
            line = ",".join([str(int(x)) for x in restored_signal])
            f.write(line + "\n")

    with open(f"{SYNTH_ROOT}/{sdm}/signals_noisy/signal_noisy_S_{s}_K_{mask_length}_I_{arrays}.csv", "w") as f_noisy:
        for restored_noisy_signal in restored_noisy_signals:
            line = ",".join([str(int(x)) for x in restored_noisy_signal])
            f_noisy.write(line + "\n")

    res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays, s, sdm, N, k=mask_length)
    return res


def read_cs(mask_length: int, arrays: int, s: int, sdm: str, N: int):
    clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv")
    indices = read_indices(f"{SYNTH_ROOT}/{sdm}/read_S_{s}_K_{mask_length}_N_{N}_I_{arrays}.csv",
                           delimiter="->")
    clean = [clean[x] for x in indices]

    clean_signals = []
    restored_signals = np.genfromtxt(f"{SYNTH_ROOT}/{sdm}/signals/signal_S_{s}_K_{mask_length}_I_{arrays}.csv", delimiter=",")
    restored_noisy_signals = np.genfromtxt(f"{SYNTH_ROOT}/{sdm}/signals_noisy/signal_noisy_S_{s}_K_{mask_length}_I_{arrays}.csv", delimiter=",")
    for i in range(arrays):
        clean_signal = np.zeros((600,))
        for ind in clean[i]:
            clean_signal[ind] = 1

        clean_signals.append(clean_signal)

    res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays, s, sdm, N, k=mask_length)
    return res


def main():
    array_nums = list(range(500, 25_001, 500))
    feature_counts = [4, 5, 6, 7, 8, 9, 10]
    N = 24_000_000

    cs_conf2_mask_lengths = [3]
    l_cs_conf2 = []
    cs_conf2_params = [(K, I, s, "cs_conf2", N) for K in cs_conf2_mask_lengths for s in feature_counts for I in array_nums]
    with mp.Pool(25) as pool:
        res = pool.starmap(read_cs, cs_conf2_params)
        l_cs_conf2 += res

    df_cs_conf2 = pd.DataFrame(l_cs_conf2)
    df_cs_conf2.to_csv(f"{SYNTH_ROOT}/cs_conf2/stats.csv", index=False)

    cs_conf1_mask_lengths = [10, 12, 14]
    l_cs_conf1 = []
    for I in array_nums:
        cs_conf1_params = [(K, I, s, "cs_conf1", N) for K in cs_conf1_mask_lengths for s in feature_counts]
        with mp.Pool(25) as pool:
            res = pool.starmap(read_cs, cs_conf1_params)
            l_cs_conf1 += res

    df_cs_conf1 = pd.DataFrame(l_cs_conf1)
    df_cs_conf1.to_csv(f"{SYNTH_ROOT}/cs_conf1/stats.csv", index=False)

    l_labels = []

    labels_params = [(3, i, s, 8_000_000) for i in array_nums for s in feature_counts]
    with mp.Pool(25) as pool:
        res = pool.starmap(check_labels, labels_params)
        l_labels += res
    df_labels = pd.DataFrame(l_labels)
    df_labels.to_csv(f"{SYNTH_ROOT}/labels/stats.csv", index=False)

    l = [*l_labels, *l_cs_conf1, *l_cs_conf2]
    sorted_l = sorted(l, key=lambda x: (x["sdm_type"], x["features_count"], x["mask_length"], x["arrays_count"]))
    df = pd.DataFrame(sorted_l)
    df = df[["sdm_type", "features_count", "mask_length", "arrays_count",
             "avg_l1", "avg_l1_noisy", "avg_fn", "avg_fn_noisy", "avg_fp", "avg_fp_noisy",
             "exact", "exact_noisy", "exact_percent", "exact_noisy_percent",
             "cells_count"]]
    df.to_csv(f"{SYNTH_ROOT}/stats.csv", index=False)


def plot():
    df = pd.read_csv(f"{SYNTH_ROOT}/stats.csv")

    l = df.to_dict("records")

    x = sorted(list({x["arrays_count"] for x in l}))

    x_ticks = np.arange(0, 25_001, 2_500)  # x
    x_tick_labels = [str(xx) for xx in x_ticks]

    features_counts = sorted(list({x["features_count"] for x in l}))

    for s in features_counts:
        # l1
        l_s = [x for x in l if x["features_count"] == s]

        cs1_10_l1 = [x["avg_l1"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_l1 = [x["avg_l1"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_l1 = [x["avg_l1"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_l1 = [x["avg_l1"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_l1 = [x["avg_l1"] for x in l_s if x["sdm_type"] == "jaeckel"]

        l1_y_ticks = np.arange(0.0, 5.0, 0.25)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in l1_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(l1_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers, [cs1_10_l1, cs1_12_l1, cs1_14_l1, cs2_l1, jaeckel_l1]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/avg_l1.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # l1 noisy
        cs1_10_l1_noisy = [x["avg_l1_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_l1_noisy = [x["avg_l1_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_l1_noisy = [x["avg_l1_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_l1_noisy = [x["avg_l1_noisy"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_l1_noisy = [x["avg_l1_noisy"] for x in l_s if x["sdm_type"] == "jaeckel"]

        l1_noisy_y_ticks = np.arange(0.0, 15.0, 0.5)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in l1_noisy_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(l1_noisy_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers, [cs1_10_l1_noisy, cs1_12_l1_noisy, cs1_14_l1_noisy, cs2_l1_noisy, jaeckel_l1_noisy]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/avg_l1_noisy.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # avg fn
        cs1_10_fn = [x["avg_fn"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_fn = [x["avg_fn"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_fn = [x["avg_fn"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_fn = [x["avg_fn"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_fn = [x["avg_fn"] for x in l_s if x["sdm_type"] == "jaeckel"]

        fn_y_ticks = np.arange(0.0, 15.0, 0.5)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in fn_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(fn_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers, [cs1_10_fn, cs1_12_fn, cs1_14_fn, cs2_fn, jaeckel_fn]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/avg_fn.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # avg fn noisy
        cs1_10_fn_noisy = [x["avg_fn_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_fn_noisy = [x["avg_fn_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_fn_noisy = [x["avg_fn_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_fn_noisy = [x["avg_fn_noisy"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_fn_noisy = [x["avg_fn_noisy"] for x in l_s if x["sdm_type"] == "jaeckel"]

        fn_noisy_y_ticks = np.arange(0.0, 15.0, 0.5)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in fn_noisy_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(fn_noisy_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers, [cs1_10_fn_noisy, cs1_12_fn_noisy, cs1_14_fn_noisy, cs2_fn_noisy, jaeckel_fn_noisy]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/avg_fn_noisy.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # avg fp
        cs1_10_fp = [x["avg_fp"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_fp = [x["avg_fp"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_fp = [x["avg_fp"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_fp = [x["avg_fp"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_fp = [x["avg_fp"] for x in l_s if x["sdm_type"] == "jaeckel"]

        fp_y_ticks = np.arange(0.0, 15.0, 0.5)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in fp_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(fp_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers, [cs1_10_fp, cs1_12_fp, cs1_14_fp, cs2_fp, jaeckel_fp]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/avg_fp.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # avg fp noisy
        cs1_10_fp_noisy = [x["avg_fp_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_fp_noisy = [x["avg_fp_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_fp_noisy = [x["avg_fp_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_fp_noisy = [x["avg_fp_noisy"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_fp_noisy = [x["avg_fp_noisy"] for x in l_s if x["sdm_type"] == "jaeckel"]

        fp_noisy_y_ticks = np.arange(0.0, 15.0, 0.5)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in fp_noisy_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(fp_noisy_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers, [cs1_10_fp_noisy, cs1_12_fp_noisy, cs1_14_fp_noisy, cs2_fp_noisy, jaeckel_fp_noisy]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/avg_fp_noisy.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # exact percent
        cs1_10_exact_percent = [x["exact_percent"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_exact_percent = [x["exact_percent"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_exact_percent = [x["exact_percent"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_exact_percent = [x["exact_percent"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_exact_percent = [x["exact_percent"] for x in l_s if x["sdm_type"] == "jaeckel"]

        exact_percent_y_ticks = np.arange(0, 101, 10)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in exact_percent_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(exact_percent_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers,
                             [cs1_10_exact_percent, cs1_12_exact_percent, cs1_14_exact_percent, cs2_exact_percent, jaeckel_exact_percent]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/exact_percent.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # exact percent noisy
        cs1_10_exact_percent_noisy = [x["exact_percent_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_12_exact_percent_noisy = [x["exact_percent_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_14_exact_percent_noisy = [x["exact_percent_noisy"] for x in l_s if x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs2_exact_percent_noisy = [x["exact_percent_noisy"] for x in l_s if x["sdm_type"] == "cs_conf2"]
        jaeckel_exact_percent_noisy = [x["exact_percent_noisy"] for x in l_s if x["sdm_type"] == "jaeckel"]

        exact_percent_noisy_y_ticks = np.arange(0, 101, 10)
        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in exact_percent_noisy_y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(exact_percent_noisy_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for marker, y in zip(markers,
                             [cs1_10_exact_percent_noisy, cs1_12_exact_percent_noisy, cs1_14_exact_percent_noisy,
                              cs2_exact_percent_noisy, jaeckel_exact_percent_noisy]):
            ys = np.array(y)

            plt.plot(x, ys,
                     # color="k"
                     )

        plt.legend(["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/exact_percent_noisy.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # l1 capacity
        cs1_10_l1_cap = [x["arrays_count"] for x in l_s if x["avg_l1"] <= 1 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_10_l1_cap = max(cs1_10_l1_cap) if cs1_10_l1_cap else 0
        cs1_12_l1_cap = [x["arrays_count"] for x in l_s if x["avg_l1"] <= 1 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_12_l1_cap = max(cs1_12_l1_cap) if cs1_12_l1_cap else 0
        cs1_14_l1_cap = [x["arrays_count"] for x in l_s if x["avg_l1"] <= 1 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs1_14_l1_cap = max(cs1_14_l1_cap) if cs1_14_l1_cap else 0
        cs2_l1_cap = [x["arrays_count"] for x in l_s if x["avg_l1"] <= 1 and x["sdm_type"] == "cs_conf2"]
        cs2_l1_cap = max(cs2_l1_cap) if cs2_l1_cap else 0
        jaeckel_l1_cap = [x["arrays_count"] for x in l_s if x["avg_l1"] <= 1 and x["sdm_type"] == "jaeckel"]
        jaeckel_l1_cap = max(jaeckel_l1_cap) if jaeckel_l1_cap else 0

        l1_cap_y_ticks = np.arange(0, 25_001, 2_500)
        y_tick_labels = [str(k) for k in l1_cap_y_ticks]
        xx = [1, 2, 3, 4, 5]
        xx_ticks = [1, 2, 3, 4, 5]
        xx_tick_labels = [1, 2, 3, 4, 5]
        # plt.xticks(xx_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(l1_cap_y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        ys = np.array([cs1_10_l1_cap, cs1_12_l1_cap, cs1_14_l1_cap, cs2_l1_cap, jaeckel_l1_cap])

        plt.bar(xx, ys)
        plt.xticks([1, 2, 3, 4, 5], ["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/l1_capacity.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # l1 noisy capacity
        cs1_10_l1_noisy_cap = [x["arrays_count"] for x in l_s if x["avg_l1_noisy"] <= 1 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_10_l1_noisy_cap = max(cs1_10_l1_noisy_cap) if cs1_10_l1_noisy_cap else 0
        cs1_12_l1_noisy_cap = [x["arrays_count"] for x in l_s if x["avg_l1_noisy"] <= 1 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_12_l1_noisy_cap = max(cs1_12_l1_noisy_cap) if cs1_12_l1_noisy_cap else 0
        cs1_14_l1_noisy_cap = [x["arrays_count"] for x in l_s if x["avg_l1_noisy"] <= 1 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs1_14_l1_noisy_cap = max(cs1_14_l1_noisy_cap) if cs1_14_l1_noisy_cap else 0
        cs2_l1_noisy_cap = [x["arrays_count"] for x in l_s if x["avg_l1_noisy"] <= 1 and x["sdm_type"] == "cs_conf2"]
        cs2_l1_noisy_cap = max(cs2_l1_noisy_cap) if cs2_l1_noisy_cap else 0
        jaeckel_l1_noisy_cap = [x["arrays_count"] for x in l_s if x["avg_l1_noisy"] <= 1 and x["sdm_type"] == "jaeckel"]
        jaeckel_l1_noisy_cap = max(jaeckel_l1_noisy_cap) if jaeckel_l1_noisy_cap else 0

        l1_noisy_cap_y_ticks = np.arange(0, 10_001, 500)
        y_tick_labels = [str(k) for k in l1_noisy_cap_y_ticks]
        xx = [1, 2, 3, 4, 5]
        # plt.xticks(xx_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(l1_noisy_cap_y_ticks, labels=y_tick_labels)
        ys = np.array([cs1_10_l1_noisy_cap, cs1_12_l1_noisy_cap, cs1_14_l1_noisy_cap, cs2_l1_noisy_cap, jaeckel_l1_noisy_cap])

        plt.bar(xx, ys)
        plt.xticks(xx, ["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/l1_noisy_capacity.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # exact percent capacity
        cs1_10_exact_percent_cap = [x["arrays_count"] for x in l_s if x["exact_percent"] >= 99 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_10_exact_percent_cap = max(cs1_10_exact_percent_cap) if cs1_10_exact_percent_cap else 0
        cs1_12_exact_percent_cap = [x["arrays_count"] for x in l_s if x["exact_percent"] >= 99 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_12_exact_percent_cap = max(cs1_12_exact_percent_cap) if cs1_12_exact_percent_cap else 0
        cs1_14_exact_percent_cap = [x["arrays_count"] for x in l_s if x["exact_percent"] >= 99 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs1_14_exact_percent_cap = max(cs1_14_exact_percent_cap) if cs1_14_exact_percent_cap else 0
        cs2_exact_percent_cap = [x["arrays_count"] for x in l_s if x["exact_percent"] >= 99 and x["sdm_type"] == "cs_conf2"]
        cs2_exact_percent_cap = max(cs2_exact_percent_cap) if cs2_exact_percent_cap else 0
        jaeckel_exact_percent_cap = [x["arrays_count"] for x in l_s if x["exact_percent"] >= 99 and x["sdm_type"] == "jaeckel"]
        jaeckel_exact_percent_cap = max(jaeckel_exact_percent_cap) if jaeckel_exact_percent_cap else 0

        l1_noisy_cap_y_ticks = np.arange(0, 25_001, 2_500)
        y_tick_labels = [str(k) for k in l1_noisy_cap_y_ticks]
        xx = [1, 2, 3, 4, 5]
        # plt.xticks(xx_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(l1_noisy_cap_y_ticks, labels=y_tick_labels)
        ys = np.array([cs1_10_exact_percent_cap, cs1_12_exact_percent_cap, cs1_14_exact_percent_cap, cs2_exact_percent_cap, jaeckel_exact_percent_cap])

        plt.bar(xx, ys)
        plt.xticks(xx, ["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/exact_percent_capacity.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")

        # exact percent noisy capacity
        cs1_10_exact_percent_noisy_cap = [x["arrays_count"] for x in l_s if x["exact_percent_noisy"] >= 99 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 10]
        cs1_10_exact_percent_noisy_cap = max(cs1_10_exact_percent_noisy_cap) if cs1_10_exact_percent_noisy_cap else 0
        cs1_12_exact_percent_noisy_cap = [x["arrays_count"] for x in l_s if x["exact_percent_noisy"] >= 99 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 12]
        cs1_12_exact_percent_noisy_cap = max(cs1_12_exact_percent_noisy_cap) if cs1_12_exact_percent_noisy_cap else 0
        cs1_14_exact_percent_noisy_cap = [x["arrays_count"] for x in l_s if x["exact_percent_noisy"] >= 99 and x["sdm_type"] == "cs_conf1" and x["mask_length"] == 14]
        cs1_14_exact_percent_noisy_cap = max(cs1_14_exact_percent_noisy_cap) if cs1_14_exact_percent_noisy_cap else 0
        cs2_exact_percent_noisy_cap = [x["arrays_count"] for x in l_s if x["exact_percent_noisy"] >= 99 and x["sdm_type"] == "cs_conf2"]
        cs2_exact_percent_noisy_cap = max(cs2_exact_percent_noisy_cap) if cs2_exact_percent_noisy_cap else 0
        jaeckel_exact_percent_noisy_cap = [x["arrays_count"] for x in l_s if x["exact_percent_noisy"] >= 99 and x["sdm_type"] == "jaeckel"]
        jaeckel_exact_percent_noisy_cap = max(jaeckel_exact_percent_noisy_cap) if jaeckel_exact_percent_noisy_cap else 0

        l1_noisy_cap_y_ticks = np.arange(0, 5_001, 500)
        y_tick_labels = [str(k) for k in l1_noisy_cap_y_ticks]
        xx = [1, 2, 3, 4, 5]
        # plt.xticks(xx_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(l1_noisy_cap_y_ticks, labels=y_tick_labels)
        ys = np.array([cs1_10_exact_percent_noisy_cap, cs1_12_exact_percent_noisy_cap, cs1_14_exact_percent_noisy_cap, cs2_exact_percent_noisy_cap, jaeckel_exact_percent_noisy_cap])

        plt.bar(xx, ys)
        plt.xticks(xx, ["CS1, K=10", "CS1, K=12", "CS1, K=14", "CS2, K=3", "Jaeckel, K=3"])

        l1_plot_path = f"{SYNTH_ROOT}/plots/s{s}/exact_percent_noisy_capacity.png"
        plt.savefig(l1_plot_path)
        plt.close()
        print(f"Image {l1_plot_path} was saved")


if __name__ == "__main__":
    # main()
    plot()
