import numc as nc
import dumbpy as db
from unittests.utils import *

"""
2. We are NOT testing operations on 1D matrices because their shapes are tricky. However, slicing on 1D matrices is expected to work. Moreover, 1D matrices' shapes are length-1 tuples. For example, suppose a is a 1D matrix with 3 elements, then its shape should be (3,). When slicing 1D matrices, ONLY integers and 1 slice are valid keys. This means that operations like a[0:1, 0:1] is not valid.
3. The indexing section has been updated in the spec!

Instead of key being int, slice, or a tuple of slices, 
it can now be int, slice, (int, slice), (slice, int), (slice, slice), or (int, int). 

4. For Matrix61c_set_subscript, if the resulting slice is 2D, 
and value is a list but not 2D list, throw a value error.

5. We expect you to handle negative indices in slices. 
This is because PySlice_GetIndicesEx should already handle 
negative indices in slices for you. However, 
we do NOT expect you to handle negative indexing using integers. 
This means for all "int" in a[int, slice], a[slice, int], a[int, int], and a[int], 
negative indexing is NOT allowed.

6. Throw a value error if you are trying to initialize matrices 
with non-positive dimensions.

7. When slicing 1D matrices, whether the matrix 
has 1 row or 1 column does not matter. 
Please find more information on the expected behavior in the spec in the 
description for Matrix61c_subscript.
"""


dp_mat1, nc_mat1 = rand_dp_nc_matrix(3, 3, seed=0)
dp_mat2, nc_mat2 = rand_dp_nc_matrix(3, 3, seed=1)

#dp_mat, nc_mat = rand_dp_nc_matrix(4, 3, seed=1)



#Ask on piazza about these weird edge cases
#with self.assertRaises(TypeError):

try:
    #Resulting slice is 1D, but v has the wrong length, or if any element of v is not a float or int.
    nc_mat1[1:11000]
except ValueError:
    print("whoopsies ncsdf error")

#dp_mat[1, None]
#dp_mat[None, 1]

#print(dp_mat[dp_mat, 1])
#print(dp_mat[None, 1])
#print(nc_mat[nc_mat, 1])
#print(nc_mat[None, 1])

#with self.assertRaises(ValueError):
#    nc_mat[1, nc_mat]

"""
try:
    dp_mat[2:4, :] = [[12, 123, 2], [1234, 123, 234]]
except ValueError:
    print("whoopsies dp error")

try:
    nc_mat[2:4, :] = [[12, 123, 2], [1234, 123, 234]]
except ValueError:
    print("whoopsies nc error")

print(dp_mat)
print(nc_mat)
"""