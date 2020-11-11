from utils import *
from unittest import TestCase

"""
For each operation, you should write tests to test  on matrices of different sizes.
Hint: use dp_mc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""

### Global variables
fuzz = 10
fuzz_rep = 10
scale = 4
### DANGEROUS CHANGE WITH CAUTION

class TestAdd(TestCase):
    def test_small_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(10, 100, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(10, 100, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(11023, 1241, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(11023, 1241, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_fractional_scaling_add(self):
        print()
        for n in range(1, scale):
            dp_mat1, nc_mat1 = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=0)
            dp_mat2, nc_mat2 = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=1)
            is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_fuzz_add(self):
        print()
        for n in range(fuzz_rep): 
            row = np.random.randint(1, fuzz) 
            col = np.random.randint(1, fuzz)
            dp_mat1, nc_mat1 = rand_dp_nc_matrix(row, col, seed=0)
            dp_mat2, nc_mat2 = rand_dp_nc_matrix(row, col, seed=1)
            is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_incorrect_dimension_add(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 4, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(4, 2, seed=1)
        self.assertRaises(ValueError, compute, [dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")

    def test_incorrect_type_add(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        self.assertRaises(TypeError, compute, [dp_mat1, dp_mat2], [nc_mat1, 3], "add")

class TestSub(TestCase):
    def test_small_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(10, 100, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(10, 100, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(11023, 1241, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(11023, 1241, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_fractional_scaling_sub(self):
        print()
        for n in range(1, scale): 
            dp_mat1, nc_mat1 = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=0)
            dp_mat2, nc_mat2 = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=1)
            is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_fuzz_sub(self):
        print()
        for n in range(fuzz_rep):
            row = np.random.randint(1, fuzz)
            col = np.random.randint(1, fuzz)
            dp_mat1, nc_mat1 = rand_dp_nc_matrix(row, col, seed=0)
            dp_mat2, nc_mat2 = rand_dp_nc_matrix(row, col, seed=1)
            is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_incorrect_dimension_sub(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 4, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(4, 2, seed=1)
        self.assertRaises(ValueError, compute, [dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")

    def test_incorrect_type_sub(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        self.assertRaises(TypeError, compute, [dp_mat1, dp_mat2], [nc_mat1, 3], "sub")

class TestAbs(TestCase):
    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(123, 100, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1230, 1200, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_fractional_scaling_abs(self):
        print()
        for n in range(1,scale): 
            dp_mat, nc_mat = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=0)
            is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_fuzz_abs(self):
        print()
        for n in range(fuzz_rep): 
            row = np.random.randint(1, fuzz) 
            col = np.random.randint(1, fuzz)
            dp_mat, nc_mat = rand_dp_nc_matrix(row, col, seed=0)
            is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

class TestNeg(TestCase):
    def test_small_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(243, 123, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2434, 1223, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_fractional_scaling_neg(self):
        print()
        for n in range(1,scale): 
            dp_mat, nc_mat = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=0)
            is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_fuzz_neg(self):
        print()
        for n in range(fuzz_rep):
            row = np.random.randint(1, fuzz)
            col = np.random.randint(1, fuzz)
            dp_mat, nc_mat = rand_dp_nc_matrix(row, col, seed=0)
            is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

class TestMul(TestCase):
    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
    
    def test_medium_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(123, 100, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(100, 321, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1223, 1100, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1100, 2321, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_fractional_scaling_mul(self):
        print()
        for n in range(1, scale): 
            dp_mat1, nc_mat1 = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=0)
            dp_mat2, nc_mat2 = rand_dp_nc_matrix(10 ** n, 10 ** n, seed=1)
            is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

    def test_fuzz_mul(self):
        print()
        for n in range(fuzz_rep):
            x = np.random.randint(1, fuzz)
            y = np.random.randint(1, fuzz)
            dp_mat1, nc_mat1 = rand_dp_nc_matrix(x, y, seed=0)
            dp_mat2, nc_mat2 = rand_dp_nc_matrix(y, x, seed=1)
            is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
            self.assertTrue(is_correct)
            print_speedup(speed_up)

class TestPow(TestCase):
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat, 3], [nc_mat, 3], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat, 10], [nc_mat, 10], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat, 50], [nc_mat, 50], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    #Come up with more clever tests later

class TestGet(TestCase):
    def test_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEquals(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestShape(TestCase):
    def test_shape(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)
