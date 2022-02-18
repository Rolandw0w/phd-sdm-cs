from datetime import datetime

from matplotlib import pyplot as plt
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


def g(path, dtype=np.int8, delimiter=None):
    with open(path, "r") as f:
        content = f.read()
        strings = content.split("\n")
        res = []
        for string in strings:
            if isinstance(delimiter, str):
                xx = string.split(",")
            else:
                xx = string
            if len(xx) != 600:
                continue
            arr = [dtype(x) for x in xx]
            np_arr = np.array(arr, dtype=dtype)
            res.append(np_arr)
        return res


def get_jaeckel(N):
    jaeckel_root = SYNTH_ROOT / "jaeckel"
    try:
        df = pd.read_csv(jaeckel_root / "jaeckel_small.csv")
    except FileNotFoundError:
        records = []
        for s in s_s:
            clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv", type_=np.int16)
            clean_signals = []
            for arrays_count in list(range(20_000, 100_000 + 1, 20_000)):
                print(datetime.utcnow(), f"Started Jaeckel (s={s}, arrays_count={arrays_count})")

                restored_signals = g(jaeckel_root / f"s{s}/read_K_4_I_{arrays_count}.csv", delimiter=",", dtype=np.int8)
                # restored_signals = np.genfromtxt(jaeckel_root / f"s{s}/read_K_4_I_{arrays_count}.csv", dtype=np.int8, delimiter="//")
                restored_noisy_signals = g(jaeckel_root / f"s{s}/read_noisy_K_4_I_{arrays_count}.csv", delimiter=",", dtype=np.int8)

                for i in range(arrays_count):
                    clean_signal = np.zeros((600,), dtype=np.int8)
                    for ind in clean[i]:
                        clean_signal[ind] = 1
                    clean_signals.append(clean_signal)
                res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays_count, s, "jaeckel", N, k=4)
                records.append(res)
        df = pd.DataFrame(records)
        df.to_csv(jaeckel_root / "jaeckel_small.csv", index=False)

    return df


def get_jaeckel_k(N):
    jaeckel_root = SYNTH_ROOT / "jaeckel"
    try:
        df = pd.read_csv(jaeckel_root / "jaeckel_comp.csv")
    except FileNotFoundError:
        records = []
        Ks = [3, 4, 5]
        for s in [16]:
            for K in Ks:
                clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv", type_=np.int16)
                clean_signals = []
                for arrays_count in list(range(500_000, 2_000_000 + 1, 500_000)):
                    print(datetime.utcnow(), f"Started Jaeckel (s={s}, K={K}, arrays_count={arrays_count})")

                    restored_signals = g(jaeckel_root / f"s{s}/read_K_{K}_I_{arrays_count}.csv", dtype=np.int8)
                    # restored_signals = np.genfromtxt(jaeckel_root / f"s{s}/read_K_4_I_{arrays_count}.csv", dtype=np.int8, delimiter="//")
                    restored_noisy_signals = g(jaeckel_root / f"s{s}/read_noisy_K_{K}_I_{arrays_count}.csv", dtype=np.int8)

                    for i in range(arrays_count):
                        clean_signal = np.zeros((600,), dtype=np.int8)
                        for ind in clean[i]:
                            clean_signal[ind] = 1
                        clean_signals.append(clean_signal)
                    res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays_count, s, "jaeckel", N, k=K)
                    records.append(res)
        df = pd.DataFrame(records)
        df.to_csv(jaeckel_root / "jaeckel_comp.csv", index=False)

    return df


def get_kanerva(N):
    kanerva_root = SYNTH_ROOT / "kanerva"
    try:
        df = pd.read_csv(kanerva_root / "kanerva_small.csv")
    except FileNotFoundError:
        records = []
        for s in s_s:
            clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv", type_=np.int16)
            clean_signals = []
            for arrays_count in list(range(20_000, 100_000 + 1, 20_000)):
                print(datetime.utcnow(), f"Started Kanerva (s={s}, arrays_count={arrays_count})")

                restored_signals = g(kanerva_root / f"s{s}/read_R_{s-4}_I_{arrays_count}.csv", dtype=np.int8)
                # restored_signals = np.genfromtxt(kanerva_root / f"s{s}/read_K_4_I_{arrays_count}.csv", dtype=np.int8, delimiter="//")
                restored_noisy_signals = g(kanerva_root / f"s{s}/read_noisy_R_{s-4}_I_{arrays_count}.csv", dtype=np.int8)

                for i in range(arrays_count):
                    clean_signal = np.zeros((600,), dtype=np.int8)
                    for ind in clean[i]:
                        clean_signal[ind] = 1
                    clean_signals.append(clean_signal)
                res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays_count, s, "kanerva", N, k=4)
                records.append(res)
        df = pd.DataFrame(records)
        df.to_csv(kanerva_root / "kanerva_small.csv", index=False)

    return df


def get_kanerva_r(N):
    kanerva_root = SYNTH_ROOT / "kanerva"
    try:
        df = pd.read_csv(kanerva_root / "kanerva_comp.csv")
    except FileNotFoundError:
        records = []
        for s in [16]:
            for radius in [11, 12, 13]:
                clean = read_arrays(f"/home/rolandw0w/Development/PhD/data/sparse_arrays/arr_{s}.csv", type_=np.int16)
                clean_signals = []
                for arrays_count in list(range(500_000, 2_000_000 + 1, 500_000)):
                    print(datetime.utcnow(), f"Started Kanerva (s={s}, arrays_count={arrays_count})")

                    restored_signals = g(kanerva_root / f"s{s}/read_R_{radius}_I_{arrays_count}.csv", dtype=np.int8)
                    # restored_signals = np.genfromtxt(kanerva_root / f"s{s}/read_K_4_I_{arrays_count}.csv", dtype=np.int8, delimiter="//")
                    restored_noisy_signals = g(kanerva_root / f"s{s}/read_noisy_R_{radius}_I_{arrays_count}.csv", dtype=np.int8)

                    for i in range(arrays_count):
                        clean_signal = np.zeros((600,), dtype=np.int8)
                        for ind in clean[i]:
                            clean_signal[ind] = 1
                        clean_signals.append(clean_signal)
                    res = _check(clean_signals, restored_signals, restored_noisy_signals, arrays_count, s, "kanerva", N, k=4)
                    res["radius"] = radius
                    records.append(res)
        df = pd.DataFrame(records)
        df.to_csv(kanerva_root / "kanerva_comp.csv", index=False)

    return df


# get_jaeckel(15_000_000)
# get_kanerva(15_000_000)
get_jaeckel_k(15_000_000)
# sdm_type = "all_synth"
# df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4_lp.csv")
# records = df.to_dict("records")
#
# jaeckel_df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/jaeckel.csv")
# jaeckel_records = jaeckel_df.to_dict("records")
#
# kanerva_df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/kanerva.csv")
# kanerva_records = kanerva_df.to_dict("records")
#
# records = df.to_dict("records")
# print()
#
# s_list = sorted(list(set([x["features_count"] for x in records])))
#
# x = list(range(500, 25_001, 500))
# x_labels = list(range(0, 25_001, 2_500))
# for s in s_list:
#     s_records = [x for x in records if x["features_count"] == s]
#     coef_N_pairs = sorted(list(set([(x["coefficient"], x["cells_count"]) for x in s_records if x["coefficient"] != 6])), key=lambda x: x[0])
#
#     kanerva = [x for x in kanerva_records if x["features_count"] == s]
#     jaeckel = [x for x in jaeckel_records if x["features_count"] == s]
#
#     for field in [
#         "avg_fn", "avg_fn_noisy",
#         "avg_fp", "avg_fp_noisy",
#         "avg_l1", "avg_l1_noisy",
#         "exact_percent", "exact_noisy_percent",
#     ]:
#         y_kanerva = [x[field] for x in kanerva]
#         plt.plot(x, y_kanerva)
#
#         y_jaeckel = [x[field] for x in jaeckel]
#         plt.plot(x, y_jaeckel)
#
#         for coef, N in coef_N_pairs:
#             s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef]
#             y = [x[field] for x in s_N_records]
#             plt.plot(x, y)
#             plt.xticks(x_labels, labels=[str(xx) for xx in x_labels], fontsize=6)
#
#         plt.legend(["Kanerva", "Jaeckel"] + [f"CS SDM ({coef}*s)" for coef, N in coef_N_pairs])
#         img_path = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}/s{s}/{field}.png"
#         plt.savefig(img_path)
#         plt.close()
#         print(f"Saved {img_path}")
#
#     # l1 capacity
#     x_l1_cap = list(range(1, len(coef_N_pairs) + 1 + 2))
#     x_labels_l1_cap = ["Kanerva", "Jaeckel"] + [f"CS SDM ({coef}*s)" for coef, N in coef_N_pairs]
#     y_l1_cap = []
#     for coef, N in coef_N_pairs:
#         s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef]
#         arrs = [x["arrays_count"] for x in s_N_records if x["avg_l1"] <= 1]
#         l1_capacity = max(arrs) if arrs else 0
#
#         y_l1_cap.append(l1_capacity)
#
#     jaeckel_arrs = [x["arrays_count"] for x in jaeckel if x["avg_l1"] <= 1]
#     jaeckel_l1_cap = max(jaeckel_arrs) if len(jaeckel_arrs) > 0 else 0
#
#     kanerva_arrs = [x["arrays_count"] for x in kanerva if x["avg_l1"] <= 1]
#     kanerva_l1_cap = max(kanerva_arrs) if len(kanerva_arrs) > 0 else 0
#
#     y_l1_cap = [kanerva_l1_cap, jaeckel_l1_cap] + y_l1_cap
#
#     plt.bar(x_l1_cap, y_l1_cap)
#     plt.xticks(x_l1_cap, x_labels_l1_cap, fontsize=5.5)
#     img_path = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}/s{s}/l1_capacity.png"
#     plt.savefig(img_path)
#     plt.close()
#
#     # l1 noisy capacity
#     x_l1_noisy_cap = list(range(1, len(coef_N_pairs) + 1 + 2))
#     x_labels_l1_noisy_cap = ["Kanerva", "Jaeckel"] + [f"{coef}*s,N={N//1_000_000}mln" for coef, N in coef_N_pairs]
#     y_l1_noisy_cap = []
#     for coef, N in coef_N_pairs:
#         s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef]
#         arrs = [x["arrays_count"] for x in s_N_records if x["avg_l1_noisy"] <= 1]
#         l1_noisy_capacity = max(arrs) if arrs else 0
#
#         y_l1_noisy_cap.append(l1_noisy_capacity)
#
#     jaeckel_arrs = [x["arrays_count"] for x in jaeckel if x["avg_l1_noisy"] <= 1]
#     jaeckel_l1_noisy_cap = max(jaeckel_arrs) if len(jaeckel_arrs) > 0 else 0
#
#     kanerva_arrs = [x["arrays_count"] for x in kanerva if x["avg_l1_noisy"] <= 1]
#     kanerva_l1_noisy_cap = max(kanerva_arrs) if len(kanerva_arrs) > 0 else 0
#
#     y_l1_noisy_cap = [kanerva_l1_noisy_cap, jaeckel_l1_noisy_cap] + y_l1_noisy_cap
#
#     plt.bar(x_l1_noisy_cap, y_l1_noisy_cap)
#     plt.xticks(x_l1_noisy_cap, x_labels_l1_noisy_cap, fontsize=5.5)
#     img_path = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}/s{s}/l1_noisy_capacity.png"
#     plt.savefig(img_path)
#     plt.close()
#
#     # exact percent capacity
#     x_exact_percent_cap = list(range(1, len(coef_N_pairs) + 1 + 2))
#     x_labels_exact_percent_cap = ["Kanerva", "Jaeckel"] + [f"CS SDM ({coef}*s)" for coef, N in coef_N_pairs]
#     y_exact_percent_cap = []
#     for coef, N in coef_N_pairs:
#         s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef]
#         arrs = [x["arrays_count"] for x in s_N_records if x["exact_percent"] >= 99]
#         exact_percent_capacity = max(arrs) if arrs else 0
#
#         y_exact_percent_cap.append(exact_percent_capacity)
#
#     jaeckel_arrs = [x["arrays_count"] for x in jaeckel if x["exact_percent"] >= 99]
#     jaeckel_exact_percent_cap = max(jaeckel_arrs) if len(jaeckel_arrs) > 0 else 0
#
#     kanerva_arrs = [x["arrays_count"] for x in kanerva if x["exact_percent"] >= 99]
#     kanerva_exact_percent_cap = max(kanerva_arrs) if len(kanerva_arrs) > 0 else 0
#
#     y_exact_percent_cap = [kanerva_exact_percent_cap, jaeckel_exact_percent_cap] + y_exact_percent_cap
#
#     plt.bar(x_exact_percent_cap, y_exact_percent_cap)
#     plt.xticks(x_exact_percent_cap, x_labels_exact_percent_cap, fontsize=5.5)
#     img_path = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}/s{s}/exact_percent_capacity.png"
#     plt.savefig(img_path)
#     plt.close()
#
#     # exact percent capacity
#     x_exact_percent_noisy_cap = list(range(1, len(coef_N_pairs) + 1 + 2))
#     x_labels_exact_percent_noisy_cap = ["Kanerva", "Jaeckel"] + [f"CS SDM ({coef}*s)" for coef, N in coef_N_pairs]
#     y_exact_percent_noisy_cap = []
#     for coef, N in coef_N_pairs:
#         s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef]
#         arrs = [x["arrays_count"] for x in s_N_records if x["exact_noisy_percent"] >= 99]
#         exact_percent_noisy_capacity = max(arrs) if arrs else 0
#
#         y_exact_percent_noisy_cap.append(exact_percent_noisy_capacity)
#
#     jaeckel_arrs = [x["arrays_count"] for x in jaeckel if x["exact_noisy_percent"] >= 99]
#     jaeckel_exact_percent_noisy_cap = max(jaeckel_arrs) if len(jaeckel_arrs) > 0 else 0
#
#     kanerva_arrs = [x["arrays_count"] for x in kanerva if x["exact_noisy_percent"] >= 99]
#     kanerva_exact_percent_noisy_cap = max(kanerva_arrs) if len(kanerva_arrs) > 0 else 0
#
#     y_exact_percent_noisy_cap = [kanerva_exact_percent_noisy_cap, jaeckel_exact_percent_noisy_cap] + y_exact_percent_noisy_cap
#
#     plt.bar(x_exact_percent_noisy_cap, y_exact_percent_noisy_cap)
#     plt.xticks(x_exact_percent_noisy_cap, x_labels_exact_percent_noisy_cap, fontsize=5.5)
#     img_path = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}/s{s}/exact_percent_noisy_capacity.png"
#     plt.savefig(img_path)
#     plt.close()
