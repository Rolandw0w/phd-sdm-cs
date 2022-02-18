import multiprocessing as mp
import os
import random

import amply
import numpy as np
from itertools import combinations
from datetime import datetime
from time import sleep


def fact(n:int):
    if n == 1:
        return 1
    return n * fact(n - 1)


def c(n: int, k: int) -> int:
    return fact(n)//(fact(k) * fact(n - k))


def generate_sparse_arrays(s: int, m: int, n: int, seed: int = 0, output_file=None) -> set:
    def _gen_arr() -> list:
        l = []
        for _ in range(s):
            r = random.randint(0, m - 1)
            while r in l:
                r = random.randint(0, m - 1)
            l.append(r)
        sorted_l = sorted(l)
        return sorted_l

    random.seed(seed)

    cc = c(m, s)

    sa_set = set()
    sss = len(sa_set)

    while sss != n and len(sa_set) != cc:
        arr = _gen_arr()
        arr_tup = tuple(arr)
        while arr_tup in sa_set:
            arr = _gen_arr()
            arr_tup = tuple(arr)
        sa_set.add(arr_tup)
        sss = len(sa_set)

    l = list(sa_set)
    for i in range(n):
        index = i % len(l)
        arr_tup = l[index]
        if output_file:
            line = ",".join([str(x) for x in arr_tup])
            output_file.write(line)
            output_file.write("\n")

    print(f"Finished s={s}")
    return sa_set


def to_binary(n, length):
    return ''.join(str(1 & int(n) >> i) for i in range(length)[::-1])


def get_mask_indices_bin(mask_length: int, addr: int, cells_count: int, seed: int = 0, output_file=None, proc=None):
    def _gen_arr() -> list:
        l = []
        for _ in range(mask_length):
            r = random.randint(0, addr - 1)
            while r in l:
                r = random.randint(0, addr - 1)
            l.append(r)
        sorted_l = sorted(l)
        return sorted_l

    sample = np.array(list(range(600)), dtype=np.int16)

    def _gen_arr_np() -> np.array:
        chosen = np.random.choice(sample, size=mask_length, replace=False)
        chosen_sorted = np.sort(chosen)
        return chosen_sorted

    random.seed(seed)
    np.random.seed(seed)

    cc = c(addr, mask_length)

    def _a(arrr, K, N, pr=False):
        l = np.zeros((addr,), dtype=np.int8)
        l[arrr] = 1
        s = np.array_split(l, 75)
        pbs = [np.packbits(ss)[0] for ss in s]
        # b = [0] * 75
        # for i, ss in enumerate(s):
        #     for j, sss in enumerate(reversed(ss)):
        #         b[i] += np.uint8(sss*(2**j))
        # line2 = [int(x).to_bytes(1, "big") for x in b]
        line = [x.tobytes() for x in pbs]
        return b"".join(line)

    if cc <= cells_count:
        indices = list(range(addr))
        combs = [comb for comb in combinations(indices, mask_length)]
    else:
        sa_set = set()
        uq = len(sa_set)
        while uq != cells_count and uq != cc:
            arr = _gen_arr_np()
            lll = _a(arr, mask_length, cells_count, pr=uq == 0)
            while lll in sa_set:
                arr = _gen_arr_np()
                lll = _a(arr, mask_length, cells_count)
            # s = len(arr.nonzero()[0])
            # aaa = np.frombuffer(lll, dtype=np.int8)
            # aaaa = [to_binary(xxx, 8) for xxx in aaa]
            sa_set.add(lll)

            uq = len(sa_set)
            if uq % 10_000_000 == 0:
                print("\t", datetime.utcnow(), mask_length, cells_count, f"{uq}/{cells_count}")

        combs = list(sa_set)
        del sa_set
        combs = sorted(combs)
        #print("\t", mask_length, cells_count, combs[0])

    ccc = 0
    d = cells_count // len(combs)

    while ccc < d:
        bytes_list = b"".join(combs)
        # for index, line in enumerate(combs):
        #     bytes_list += line
        #     # output_file.write(b"".join(line))
        #     if index % 100_000 == 0:
        #         print("\t", datetime.utcnow(), f"{index}/{cells_count}")
        del combs
        output_file.write(bytes(bytes_list))
        ccc += 1

    return


def get_mask_indices(mask_length: int, addr: int, cells_count: int, seed: int = 0, output_file=None, proc=None):
    def _gen_arr() -> list:
        l = []
        for _ in range(mask_length):
            r = random.randint(0, addr - 1)
            while r in l:
                r = random.randint(0, addr - 1)
            l.append(r)
        sorted_l = sorted(l)
        return sorted_l

    sample = np.array(list(range(600)), dtype=np.int16)

    def _gen_arr_np() -> np.array:
        chosen = np.random.choice(sample, size=mask_length, replace=False)
        chosen_sorted = tuple(np.sort(chosen))
        return chosen_sorted

    random.seed(seed)
    np.random.seed(seed)

    cc = c(addr, mask_length)

    def _a(arrr: np.ndarray):
        return arrr.tobytes()

    if cc <= cells_count:
        indices = list(range(addr))
        combs = [comb for comb in combinations(indices, mask_length)]
    else:
        sa_set = set()
        uq = len(sa_set)
        while uq != cells_count and uq != cc:
            arr = _gen_arr_np()
            while arr in sa_set:
                arr = _gen_arr_np()
            # s = len(arr.nonzero()[0])
            # aaa = np.frombuffer(lll, dtype=np.int8)
            # aaaa = [to_binary(xxx, 8) for xxx in aaa]
            sa_set.add(arr)

            uq = len(sa_set)
            if uq % 10_000_000 == 0:
                print("\t", datetime.utcnow(), mask_length, cells_count, f"{uq}/{cells_count}")

        combs = list(sa_set)
        del sa_set
        combs = sorted(combs)

    combs_str = "\n".join([",".join(map(str, comb)) for comb in combs])
    output_file.write(combs_str)

    return


def a(k, L, N, s, proc=None, mode="w"):
    sleep(N/10_000_000)
    ext = ".csv" if mode == "w" else ".b"
    path = f"../data/masks/indices_addr_{L}_N_{N}_K_{k}" + ext
    print(datetime.utcnow(), f"L={L} N={N} K={k} started")
    try:
        file_size = os.path.getsize(path)
        if file_size != 75 * N:
            raise OSError()
    except OSError:
        with open(path, mode) as f:
            get_mask_indices(k, L, N, output_file=f, seed=N, proc=proc)
            print(datetime.utcnow(), f"L={L} N={N} K={k} finished")
    else:
        print(datetime.utcnow(), f"L={L} N={N} K={k} exists and consistent")


def main():
    l = []
    for N in [40]:#, 60, 80, 100]:
        l.append((4, 600, N * 1_000_000, None, 1, "w"))
            # a(K, 600, N * 1_000_000, None, proc=1, mode="wb")
    with mp.Pool(1) as pool:
        pool.starmap(a, l)
    # a(6, 600, 15 * 1_000_000, None)
    # a(12, 600, 15 * 1_000_000, None)
    # a(16, 600, 15 * 1_000_000, None)
    # for s in [20]:#, 5, 6, 7, 8, 9, 10]:
    #     with open(f"../data/sparse_arrays/arr_{s}.csv", "w") as f:
    #         generate_sparse_arrays(s, 600, 10_000_000, output_file=f)
    # return
    # params = []
    # for k in [3]:
    #     for N in range(25, 121, 5):
    #         params.append((k, 600, N * 1_000_000, None))
    # print(len(params))
    # # for m, N in [(100, 36), (75, 45), (60, 52)]:
    # #     params.append((3, 600, N * 1_000_000, None))
    # #
    # with mp.Pool(3) as pool:
    #     pool.starmap(a, params)

    # l = [
    #     [4, [[16, 55], [14, 65], [12, 75], [10, 85], [8, 95], [6, 105], [4, 115]]],
    #     [5, [[16, 45], [14, 51], [12, 57], [10, 63], [8, 69], [6, 75], [4, 81]]],
    #     [6, [[16, 37], [14, 43], [12, 49], [10, 55], [8, 61], [6, 67], [4, 73]]],
    #     [7, [[16, 32], [14, 36], [12, 40], [10, 44], [8, 48], [6, 52], [4, 56]]],
    #     [8, [[16, 28], [14, 32], [12, 36], [10, 40], [8, 44], [6, 48], [4, 52]]],
    #     [9, [[16, 25], [14, 27], [12, 29], [10, 31], [8, 33], [6, 35], [4, 37]]],
    #     [10, [[16, 23], [14, 25], [12, 27], [10, 29], [8, 31], [6, 33], [4, 35]]],
    # ]
    # mask_lengths = [3]
    # params = []
    # for mask_length in mask_lengths:
    #     for num_ones, v1 in l:
    #         for coef, N in v1:
    #             if coef > 6:
    #                 continue
    #             params.append((mask_length, 600, N * 1_000_000, num_ones))
    #
    # # for param in params:
    # # a(*params[4])
    # i = 0
    # while i < len(params):
    #     with mp.Pool(4) as pool:
    #         pool.starmap(a, params[i:i+5])
    #         i += 5
    # with mp.Pool(10) as pool:
    #     pool.starmap(a, [(x, 600) for x in range(6, 17)])
    # for s in [4, 5, 6, 7, 8, 9, 10]:
    #     with open(f"../data/sparse_arrays/arr_{s}_{4*s}.csv", "w") as f:
    #         generate_sparse_arrays(s, 4*s, 100_000, output_file=f)


if __name__ == "__main__":
    main()
