import numc as nc
import dumbpy as db
from unittests.utils import *


dp_mat, nc_mat = rand_dp_nc_matrix(5, 5, seed=0)


dp_mat[1:3, 2:4] = [[234, 234], [23,12]]
nc_mat[1:3, 2:4] = [[234, 234], [23,12]]

print(dp_mat)
print(nc_mat)

#del a