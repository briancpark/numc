import dumbpy as dp
import numc as nc
import numpy as np
import hashlib, struct
from typing import Union, List
import operator
import time
from utils import *
import os


def has_fma():
    """
    Returns whether the CPU has FMA (through lscpu)
    """
    return "fma" in os.popen("lscpu").read()


def has_avx2():
    """
    Returns whether the CPU has AVX2 (through lscpu)
    """
    return "avx2" in os.popen("lscpu").read()


def num_cores():
    """
    Returns the number of logical cores (no hyperthreading)
    """
    cpu_info = os.popen("lscpu").read()
    for line in cpu_info.split("\n"):
        if "Core(s) per socket:" in line:
            cores_per_socket = int(line.split(":")[1].strip())
        if "Socket(s):" in line:
            sockets = int(line.split(":")[1].strip())
    return cores_per_socket * sockets


def get_cpu_freq():
    """
    Returns the CPU frequency in GHz
    """
    cpu_info = os.popen("lscpu").read()
    freq = 0
    for line in cpu_info.split("\n"):
        if "CPU max MHz:" in line:
            freq = float(line.split(":")[1].strip())
    return freq / 1000


def get_theoretical_peak():
    """
    Returns the theoretical peak performance of the CPU in GFLOPS
    """
    peak = 1

    if has_fma():
        peak *= 2

    if has_avx2():
        peak *= 4

    peak *= num_cores()

    peak *= get_cpu_freq()

    return peak


def matmul_benchmark(m, n, k):
    trials = 5
    benchmark_numpy = False

    dp_mat1, nc_mat1 = rand_dp_nc_matrix(m, k, seed=0)
    dp_mat2, nc_mat2 = rand_dp_nc_matrix(k, n, seed=1)
    a = np.random.rand(m, k)
    b = np.random.rand(k, n)

    np_times = []
    nc_times = []
    flops = 2 * m * n * k

    if benchmark_numpy:
        for i in range(trials):
            start = time.time()
            c = a @ b
            end = time.time()
            np_times.append(end - start)

        avg_time = np.mean(np_times)
        avg_flops = flops / avg_time
        print("Numpy average time: ", avg_time)
        print("Numpy average gflops: ", avg_flops / 1e9)
        print("Percent of peak: ", avg_flops / 1e9 / get_theoretical_peak() * 100)

    for i in range(trials):
        start = time.time()
        c = nc_mat1 * nc_mat2
        end = time.time()
        nc_times.append(end - start)

    avg_time = np.mean(nc_times)
    avg_flops = flops / avg_time
    print("Numc average time: ", avg_time)
    print("Numc average gflops: ", avg_flops / 1e9)
    print("Percent of peak: ", avg_flops / 1e9 / get_theoretical_peak() * 100)


if __name__ == "__main__":
    ns = [1000, 2000, 4000]
    for n in ns:
        matmul_benchmark(n, n, n)
