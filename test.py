import numc as nc
import dumbpy as db
from unittests.utils import *



#dp_mat1, nc_mat1 = rand_dp_nc_matrix(1223, 1100, seed=0)
#dp_mat2, nc_mat2 = rand_dp_nc_matrix(1100, 2321, seed=1)

dp_mat1, nc_mat1 = rand_dp_nc_matrix(10, 10, seed=0)
dp_mat2, nc_mat2 = rand_dp_nc_matrix(10, 10, seed=1)
print(nc_mat1 * nc_mat2)
print(dp_mat1 * dp_mat2)




#dp_mat, nc_mat = rand_dp_nc_matrix(1000, 1000, seed=0)


"""
#Fix this malloc problem later
for i in range(100000):
    A = dp.Matrix(10000, 10000)
    print(i)
    del A

#del a

"""