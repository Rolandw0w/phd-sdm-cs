import numpy as np


s_coef_N = [
    (12, 6, 100),
    (12, 8, 80),
    (12, 12, 50),
    (16, 6, 80),
    (16, 8, 60),
    (16, 12, 40),
    (20, 6, 60),
    (20, 8, 40),
    (20, 12, 30),
]


def print_cs_sdm():
    for s, coef, N in s_coef_N:
        m = s*coef
        with open(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s{s}/acts_m_{m}_K_4_N_{N*1_000_000}_I_100000.csv", "r") as f:
            content = f.read()
            l = list(map(int, content.split(",")[:-1]))
            avg = np.mean(l)
            zeros = len([x for x in l if x == 0])
            print(s, coef, N, avg, zeros / 1_000)


def print_kanerva(i=2_000_000):
    s_unq = sorted(list(set([s for s, _, _ in s_coef_N])))
    for s in s_unq:
        r = s - 3
        with open(f"/home/rolandw0w/Development/PhD/output/synth/kanerva/s{s}/acts_R_{r}_I_{i}.csv", "r") as f:
            content = f.read()
            l = list(map(int, content.split(",")[:-1]))
            avg = np.mean(l)
            zeros = len([x for x in l if x == 0])
            print("Kanerva", s, r, avg, zeros * 100 / i)


def print_jaeckel(i=2_000_000):
    s_unq = sorted(list(set([s for s, _, _ in s_coef_N])))
    k = 4
    for s in s_unq:
        with open(f"/home/rolandw0w/Development/PhD/output/synth/jaeckel/s{s}/acts_K_{k}_I_{i}.csv", "r") as f:
            content = f.read()
            l = list(map(int, content.split(",")[:-1]))
            avg = np.mean(l)
            zeros = len([x for x in l if x == 0])
            print("Jaeckel", s, k, avg, zeros * 100 / i)



print_kanerva()
print_jaeckel()
