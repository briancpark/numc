from distutils.core import setup, Extension
import sysconfig

def main():
	CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
	LDFLAGS = ['-fopenmp']
	# Use the setup function we imported and set up the modules.
	# You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
	# TODO: YOUR CODE HERE
    
	module = Extension('numc', 
        sources=['numc.c', 'matrix.c'], 
        extra_compile_args=CFLAGS, 
        extra_link_args=LDFLAGS, 
        include_dirs=['/usr/include/python3.6m'])

	setup(name='numc',
          version='1.0',
          description='A Python3 libary for fast matrix operations!',
          long_description='CS 61C Project 4 on Parallelism', 
          author='Brian Park and Kaelyn Kim',
          author_email='briancpark@berkeley.edu',
          url='https://cs61c.org/fa20/projects/proj4/',
          ext_modules=[module])

if __name__ == "__main__":
    main()
