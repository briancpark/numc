# numc
Kaelyn Kim, Brian Park

Here's what I did in project 4:
-

## Task 1
Implemented various functions. There were some conceptual challenges in understanding how slicing works, but turns out they're really represented as linked list of matrix structs and shifting memory pointers by offsets. Matrix operations were actually pretty straightfoward, and *much* easier compared to the matrix operations we had to implement in RISC-V assembly for project 2. Optimizations will be later applied once we learn parallelism!

### `int allocate_matrix(matrix **mat, int rows, int cols)`
The very first function we implemented. In order to understand this and the spec inside and out, we took the time to carefully implement everything before moving on to other function. An important design choice was to use a double pointer array to properly address and call on 2D matrices. It makes sense to do so, and made slicing implementation much more intuitive and less of a hassle.
### `int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset, int rows, int cols)`
Understanding `allocate_matrix()` allowed us to proceed further to this function. It turns out this was related to slicing. We had to come back later and fix bugs related to pointers. For the longest time, Brian was stuck up on how to shift pointers around and weird bugs would happen. After Brian rigorously debugged in `cgdb`, he found the pointers could be elegantly be shifted with:
```c
(*mat)->data[i] = from->data[i + row_offset] + col_offset;
```
### `void deallocate_matrix(matrix *mat)`
A bit complicated as we had to deal with how to alocate parents and child matrices, but soon realized it was really just linked list/tree data structure. We just have to recursively check on parent/child matrices if they are referenced or need to be free'd.
### `double get(matrix *mat, int row, int col)`
Just a clean one liner.
### `void set(matrix *mat, int row, int col, double val)`
Another clean one liner!
### `void fill_matrix(matrix *mat, double val)`
Very simple. Code explains itself.
### `int add_matrix(matrix *result, matrix *mat1, matrix *mat2)`
Also very simple for the naive solution. Will be explained how it was sped up in Task 4.
### `int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)`
Also very simple for the naive solution. Will be explained how it was sped up in Task 4.
### `int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)`
Naive solution was a bit trick to implement, as we realized that trying to multiply a matrix by itself would scribble data from the original matrix and overwrite it. Thus, we had to allocate another temporary array during matrix operation.
### `int pow_matrix(matrix *result, matrix *mat, int pow)`
Naive solution used just a for loop of `mul_matrix()`.
### `int neg_matrix(matrix *result, matrix *mat)`
Very simple, just multiply by -1 for each entry.
### `int abs_matrix(matrix *result, matrix *mat)`
Also very simple, but encountered a bug where the C standard `abs()` function would return an int. This was simply changed to `fabs()` (for floating point absolute value) and our bug was fixed!


## Task 2
Pretty simple, and straightforward. Skimming through the Python3 doc helped us solve the problem. Now our Python can communicate with out C code and vice versa! Woohoo!

## Task 3
For this task, a lot of the functions were very simple after truly understanding the python docs. The Python docs were confusing to read at first, but rewarding when we slowly caught up with it and make things work! 

Hardest part were slicing, just due to the sheer complexity and *MANY* different errors we needed to handle. 

We also kept failing `set()` for the most minor and funniest reason. Brian kept improving on slicing functionality, thinking that the autograder's set correctness test robustly tested on it. But turns our it was resolved in OH when Brian realized that set didn't have to be invoked through slicing. It could be invoked through `set(self, i, j, val)` in Python. The bug was just a simple error handling between `if (!PyLong_Check(val) || !PyFloat_Check(val))` to `if (!PyLong_Check(val) && !PyFloat_Check(val))` Even though time debugging it was fustrating, at least Brian made sure many slicing errors were handled and properly working. 

### Testing
For robust testing, Brian made a simple bash script to compile and run all tests under the executable 
```
./skiddie
```
It will do everything in one line, because well... Brian is lazy and a script kiddie.

## Task 4


TAs/Tutors who helped us: Kunal Dutta, Luke Mujica, Jie Qiu