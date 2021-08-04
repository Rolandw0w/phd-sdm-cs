import multiprocessing as mp
import random

from itertools import combinations


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

    while len(sa_set) != n and len(sa_set) != cc:
        arr = _gen_arr()
        arr_tup = tuple(arr)
        while arr_tup in sa_set:
            arr = _gen_arr()
            arr_tup = tuple(arr)
        sa_set.add(arr_tup)

    l = sorted(list(sa_set))
    for i in range(n):
        index = i % len(l)
        arr_tup = l[index]
        if output_file:
            line = ",".join([str(x) for x in arr_tup])
            output_file.write(line)
            output_file.write("\n")

    print(f"Finished s={s}")
    return sa_set


def get_mask_indices(mask_length: int, addr: int, cells_count: int, seed: int = 0, output_file=None):
    def _gen_arr() -> list:
        l = []
        for _ in range(mask_length):
            r = random.randint(0, addr - 1)
            while r in l:
                r = random.randint(0, addr - 1)
            l.append(r)
        sorted_l = sorted(l)
        return sorted_l

    random.seed(seed)

    cc = c(addr, mask_length)

    if cc <= cells_count:
        indices = list(range(addr))
        combs = [comb for comb in combinations(indices, mask_length)]
    else:
        sa_set = set()
        uq = len(sa_set)
        while uq != cells_count and uq != cc:
            arr = _gen_arr()
            arr_tup = tuple(arr)
            while arr_tup in sa_set:
                arr = _gen_arr()
                arr_tup = tuple(arr)
            sa_set.add(arr_tup)

            uq = len(sa_set)
            if uq % 1_000_000 == 0:
                print(f"{uq}/{cells_count}")

        combs = sorted(list(sa_set))

    ccc = 0
    lines_written = 0
    d = cells_count // len(combs)

    while ccc < d:
        for index, arr_tup in enumerate(combs):
            line = ",".join([str(x) for x in arr_tup])
            output_file.write(line)
            output_file.write("\n")
            lines_written += 1
        ccc += 1

    rem = cells_count - d*len(combs)
    if rem != 0:
        sample = random.sample(combs, rem)
        for arr_tup in sample:
            line = ",".join([str(x) for x in arr_tup])
            output_file.write(line)
            output_file.write("\n")
            lines_written += 1

    return


def get_mask_indices_it(mask_length: int, addr: int, cells_count: int, seed: int = 0, output_file=None):
    cc = c(addr, mask_length)
    skip_chance = 1 - min(cells_count/cc, 1)

    indices = list(range(addr))
    l: int = 0
    flag = True

    random.seed(seed)
    combs = []
    for index, comb in enumerate(combinations(indices, mask_length)):
        combs.append(comb)
    while flag:
        for comb in combinations(indices, mask_length):
            r = random.random()
            if r > skip_chance:
                line = ",".join([str(x) for x in comb])
                output_file.write(line + "\n")
                l += 1
                if l == cells_count:
                    flag = False
                    break


def a(k, L, N, s):
    with open(f"../data/masks/indices_addr_{L}_N_{N}_K_{k}_s_{s}.csv", "w") as f:
        print(f"L={L} N={N} K={k} started")
        get_mask_indices(k, L, N, output_file=f, seed=N)
        print(f"L={L} N={N} K={k} finished")


def main():
    l = [
        [4, [[16, 55], [14, 65], [12, 75], [10, 85], [8, 95], [6, 105], [4, 115]]],
        [5, [[16, 45], [14, 51], [12, 57], [10, 63], [8, 69], [6, 75], [4, 81]]],
        [6, [[16, 37], [14, 43], [12, 49], [10, 55], [8, 61], [6, 67], [4, 73]]],
        [7, [[16, 32], [14, 36], [12, 40], [10, 44], [8, 48], [6, 52], [4, 56]]],
        [8, [[16, 28], [14, 32], [12, 36], [10, 40], [8, 44], [6, 48], [4, 52]]],
        [9, [[16, 25], [14, 27], [12, 29], [10, 31], [8, 33], [6, 35], [4, 37]]],
        [10, [[16, 23], [14, 25], [12, 27], [10, 29], [8, 31], [6, 33], [4, 35]]],
    ]
    mask_lengths = [3]
    params = []
    for mask_length in mask_lengths:
        for num_ones, v1 in l:
            for coef, N in v1:
                if coef > 6:
                    continue
                params.append((mask_length, 600, N * 1_000_000, num_ones))

    # for param in params:
    # a(*params[4])
    i = 0
    while i < len(params):
        with mp.Pool(4) as pool:
            pool.starmap(a, params[i:i+5])
            i += 5
    # with mp.Pool(10) as pool:
    #     pool.starmap(a, [(x, 600) for x in range(6, 17)])
    # for s in [4, 5, 6, 7, 8, 9, 10]:
    #     with open(f"../data/sparse_arrays/arr_{s}_{4*s}.csv", "w") as f:
    #         generate_sparse_arrays(s, 4*s, 100_000, output_file=f)
    # for s in [4, 5, 6, 7, 8, 9, 10]:
    #     with open(f"../data/sparse_arrays/arr_{s}.csv", "w") as f:
    #         generate_sparse_arrays(s, 600, 1_000_000, output_file=f)


if __name__ == "__main__":
    main()
