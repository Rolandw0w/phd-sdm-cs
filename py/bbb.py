import multiprocessing as mp
from datetime import datetime

import numpy as np

import numba as nb
from numba import cuda, int8

from time import time


np.random.seed(0)

matrix = np.genfromtxt("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s12/matrix_96.csv", delimiter=",").astype(np.int8)
arrs = [np.random.choice([0, 1], size=(matrix.shape[1],), p=(0.98, 0.02)).astype(np.int8) for _ in range(180_000)]


def a_np(arr):
    t = time()
    m = np.matmul(matrix, arr)
    return time() - t



@nb.jit('void(int8[:,:],int8[:],int8[:])')
def matmul(matrix, arr, res_arr):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            res_arr[i] += matrix[i,j] * arr[j]


def a_nb(arr):
    t = time()
    r = np.zeros((matrix.shape[0], ), dtype=np.int8)
    matmul(matrix, arr, r)
    return time() - t


TPB = 16


@cuda.jit('void(int8[:,:], int8[:], int8[:])')
def cu_matrix_vector(matrix, arr, res_arr):
    row = cuda.grid(1)
    print(row)
    if row < matrix.shape[0]:
        s = 0
        for i in range(matrix.shape[1]):
            s += matrix[row, i] * arr[i]

        res_arr[row] = s


r = np.zeros((matrix.shape[0], ), dtype=np.int8)
dM = cuda.to_device(matrix)
dA = cuda.to_device(arrs[0])
dR = cuda.to_device(r)
cu_matrix_vector[(matrix.shape[0]+511)//512, 512](dA, dM, dR)

dR.copy_to_host()


with mp.Pool(30) as pool:
    print(datetime.utcnow(), "Started np")
    tl = pool.map(a_np, arrs)
    print(datetime.utcnow(), "Finished np")


print("np", len(tl), np.mean(tl), np.sum(tl))


with mp.Pool(30) as pool:
    print(datetime.utcnow(), "Started nb")
    tl = pool.map(a_nb, arrs)
    print(datetime.utcnow(), "Finished nb")


print("nb", len(tl), np.mean(tl), np.sum(tl))
