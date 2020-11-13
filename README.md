# numc
Kaelyn Kim, Brian Park

Here's what I did in project 4:
-

## Task 1
Implemented various functions. There were some conceptual challenges in understanding how slicing works, but turns out they're really represented as linked list of matrix structs and shifting memory pointers by offsets. Matrix operations were actually pretty straightfoward, and *much* easier compared to the matrix operations we had to implement in RISC-V assembly for project 2. Optimizations will be later applied once we learn parallelism!

### `int allocate_matrix(matrix **mat, int rows, int cols)`
### `int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset, int rows, int cols)`
### `void deallocate_matrix(matrix *mat) `
### `double get(matrix *mat, int row, int col)`
### `void set(matrix *mat, int row, int col, double val)`
### `void fill_matrix(matrix *mat, double val)`
### `int add_matrix(matrix *result, matrix *mat1, matrix *mat2)`
### `int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)`
### `int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)`
### `int pow_matrix(matrix *result, matrix *mat, int pow)`
### `int neg_matrix(matrix *result, matrix *mat)`
### `int abs_matrix(matrix *result, matrix *mat)`


## Task 2
Pretty simple, and straightforward. Skimming through the Python3 doc helped us solve the problem.

## Task 3

## Task 4

## Task 5

TAs/Tutors who helped us: Kunal Dutta, Luke Mujica, Jie Qiu