import numc as nc
import dumbpy as db
from unittests.utils import *

"""
a = nc.Matrix(3, 3)
a[0:1, 0:1] = 0.0 # Resulting slice is 1 by 1
a[:, 0] = [1, 1, 1] # Resulting slice is 1D
print(a)
#[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
a[0, :] = [2, 2, 2] # Resulting slice is 1D
print(a)
#[[2.0, 2.0, 2.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
a[0:2, 0:2] = [[1, 2], [3, 4]] # Resulting slice is 2D
print(a)
#[[1.0, 2.0, 2.0], [3.0, 4.0, 0.0], [1.0, 0.0, 0.0]]

a = nc.Matrix(2, 2)
a[0:1, 0:1] = 1.0
print(a)
#[[1.0, 0.0], [0.0, 0.0]]
a[1] = [2, 2]
print(a)
#[[1.0, 0.0], [2.0, 2.0]]
b = a[1]
b[1] = 3
print(a)
#[[1.0, 0.0], [2.0, 3.0]]
"""

a = nc.Matrix(4, 4)
b = a[0:3, 0:3]
c = b[1:3, 1:3]
c[0] = [2, 2] # Changing c should change both a and b
print(c)
#[[2.0, 2.0], [0.0, 0.0]]
print(b)
#[[0.0, 0.0, 0.0], [0.0, 2.0, 2.0], [0.0, 0.0, 0.0]]
print(a)
#[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]