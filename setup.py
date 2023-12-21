from distutils.core import setup, Extension
import os
import sysconfig


def main():
    CFLAGS = [
        "-g",
        "-Wall",
        "-std=c99",
        "-fopenmp",
        "-mavx",
        "-mfma",
        "-pthread",
        "-Ofast",
        "-ffast-math",
    ]
    LDFLAGS = ["-fopenmp"]

    module = Extension(
        "numc",
        sources=["numc.c", "matrix.c"],
        extra_compile_args=CFLAGS,
        extra_link_args=LDFLAGS,
        include_dirs=["/usr/include/python3.6m"],
    )

    setup(
        name="numc",
        version="1.0",
        description="A Python3 library for fast matrix operations!",
        long_description="CS 61C Project 4 on Parallelism",
        author="Brian Park",
        author_email="briancpark@berkeley.edu",
        url="https://cs61c.org/fa20/projects/proj4/",
        ext_modules=[module],
    )

    num_cores = os.cpu_count()
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_PROC_BIND"] = "spread"
    os.environ["OMP_NUM_THREADS"] = str(num_cores)


if __name__ == "__main__":
    main()
