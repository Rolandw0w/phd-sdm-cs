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
        # s = np.sum(data_np, axis=1).tolist()
        # ss = [(i, s[i]) for i in range(600)]
        # sss = sorted(ss, key=lambda x: -x[1])
        return data_np


def get_features_from_txt(features_bin_path, left_slice: int = None) -> np.ndarray:
    data_np = np.zeros((626, 9_000))
    with open(features_bin_path, "r") as f_read:
        content = f_read.read()
        for i, char in enumerate(content):
            index_1 = i % 626
            index_2 = i // 626
            data_np[index_1, index_2] = int(content[i])
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


def get_cifar10_img():
    def unpickle(file: str):
        import pickle
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d

    dd = {}
    ddd = {}
    for i in range(1, 6):
        path = f"/home/rolandw0w/Development/PhD/data/cifar-10/np/data_batch_{i}"
        d = unpickle(path)
        dat = d[b"data"]
        X = dat.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        from matplotlib import pyplot as plt
        fig, axes1 = plt.subplots(5,5,figsize=(3,3))
        for j in range(5):
            for k in range(5):
                i = np.random.choice(range(len(X)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[i:i+1][0])
        plt.show()

        file_names = d[b"filenames"]
        file_names = file_names#[file_name.decode().split("_")[-1] for file_name in file_names]
        fns = [file_name.decode().split("_")[-1] for file_name in file_names]
        for fn in fns:
            ddd.setdefault(fn, 0)
            ddd[fn] += 1
        for j in range(10_000):
            dd[file_names[j]] = (d[b"labels"][j], d[b"data"][j])
    l = [None] * len(dd)


    for k, v in dd.items():
        n = int(k.split(".")[0])
        l[n] = v

    print(dd)


# get_cifar10_img()