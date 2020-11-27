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
After the core parts of Task 3 was implemented, we could finally move on to testing our `numc` library and ensure that it works correctly. The Python `unittest` framework can be abused through tactical testing and Brian added fuzz, scaling and fuzz repetition global parameters to scale up the testing when needed. Parameters were catiously set and tuned as Brian almost crashed an entire Hive server for running large tests. 

For efficient TDD workflow, Brian made a simple bash script to compile and run all tests under the executable 
```
./skiddie.sh
```
It will do everything in one line, (load Python environment, clean, compile, and run unittests) because well... Brian is lazy and a script kiddie.

## Task 4

### Simple (Addition, Subtraction, Negation, Absolute)
Began first by trying to improve the performance of add. Once we knew how to accelerate add, the others could follow the same structure. The buildup to achieve 5X perfomance is shown through our acceleration of the `add_matrix()` function.

#### The Naive Solution
First, we began with the naive solution, and turns out it performs about the same as `dumbpy` as expected. 

```c
for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat1->cols; j++) {
        result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];  
    }
}
return 0;
```

#### Improving Memory Spatial Locality (Better Caching)
We soon realized a crutch to how our memory is managed. Because `mat->data` is stored in a 2D array. we naively fragmented the data across random segments of memory. To improve memory management, we went for a contiguous memory model with a 1D row-major array for the `mat->data`. It's still stored as a `(**double)`, so the functionality of the rest of the code stays intact!

Here is what the tranformation of memory allocation was changed:

This was the code before, where `malloc()` would fragment the data into separate blocks by rows:

```c
(*mat)->data = (double**) malloc(sizeof(double*) * rows);
for (int i = 0; i < rows; i++) {
    (*mat)->data[i] = (double *) calloc(cols, sizeof(double));
}
```

As you can observe, a simple `calloc()` is performed for the whole `mat->data`. Then `rows` are separately malloced to populate it with where each row starts in the row-major 1D array. This is the key to making matrix data still callable through `mat->data[i][j]`. This also improved caching and causes less misses since the memory is now contiguous and improved on spatial locality.

```c
double *arr = calloc(rows * cols, sizeof(double));
(*mat)->data = malloc(rows * sizeof(double*));
for (int i = 0; i < rows; i++) {
    (*mat)->data[i] = &(arr[i * cols]);
}
```

#### Speed it up with SIMD
Time to start speeding it up even more! We applied SIMD with [Intel Intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2&expand=2184,2183,2177,3980,5999,6006,3594,5392,956). [Linked here](https://ark.intel.com/content/www/us/en/ark/products/75122/intel-core-i7-4770-processor-8m-cache-up-to-3-90-ghz.html) is the specs of the [Hive machines](https://www.ocf.berkeley.edu/~hkn/hivemind/). Vectorization provided near 3X speedup compared to `dumbpy`. We took the hints provided in the specs on which intrinsics should be used very well. We mainly used 256 bit vector operations as those were the maximum data provided by the Hive's Intel i7 4th generation processors. We also unrolled the loops for just the minimal speed up, although we are not sure if that actually makes a difference since the compiler could actually unroll it, but did it to just stay safe!

```c
double *res_pointer = result->data[0];
double *mat1_pointer = mat1->data[0];
double *mat2_pointer = mat2->data[0];

for (int i = 0; i < mat1->rows * mat1->cols / 16 * 16; i += 16) {        
    _mm256_storeu_pd(res_pointer + i, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i, _mm256_loadu_pd(mat2_pointer + i)));
    _mm256_storeu_pd(res_pointer + i + 4, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 4), _mm256_loadu_pd(mat2_pointer + i + 4)));
    _mm256_storeu_pd(res_pointer + i + 8, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 8), _mm256_loadu_pd(mat2_pointer + i + 8)));
    _mm256_storeu_pd(res_pointer + i + 12, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 12), _mm256_loadu_pd(mat2_pointer + i + 12)));
}      
```

#### I hope the rope is... Multithreaded
Faster! We applied OpenMP, with a simple `pragma omp parallel for`. Was it really that easy though? No, as you saw in the previous iteration with SIMD, we had to stuff all the operations in one line. We did this to prevent any race conditions or false sharing that would happen with parallelization. Even though the Hive's machine have 8 threads, they are hyperthreaded, and the computer architecture is really just 4 cores. Hyperthreading makes it so that the 2 threads in a core would compete for data, so we catiously chose to make it run on 4 threads instead. This gives us a total speedup of 5X compared to `dumbpy`, sometimes 5.09X if lucky!

```c
double *res_pointer = result->data[0];
double *mat1_pointer = mat1->data[0];
double *mat2_pointer = mat2->data[0];

omp_set_num_threads(4);
#pragma omp parallel for
for (int i = 0; i < mat1->rows * mat1->cols / 16 * 16; i += 16) {        
    _mm256_storeu_pd(res_pointer + i, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i, _mm256_loadu_pd(mat2_pointer + i)));
    _mm256_storeu_pd(res_pointer + i + 4, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 4), _mm256_loadu_pd(mat2_pointer + i + 4)));
    _mm256_storeu_pd(res_pointer + i + 8, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 8), _mm256_loadu_pd(mat2_pointer + i + 8)));
    _mm256_storeu_pd(res_pointer + i + 12, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 12), _mm256_loadu_pd(mat2_pointer + i + 12)));
}      
```

#### Slicing Performance
We saw that we also didn't need to care about the perfomance on slice matrices, thus a simple NULL check was performed to switch over to the naive solution for sliced and small matrices as follows:
```c
if (mat1->rows < 16 || mat1->cols < 16 || mat1->parent != NULL || mat2->parent != NULL) {
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];  
        }
    }
    return 0;
}
```

#### Can We Do Even Better? (Conclusion)
We could certainly optimize it a bit more efficiently by carefully thinking about caches and virtual memory. But this is the best we could come with in terms of performance. 


### Multiply


### Power


### About the Hive CPUs
Will be useful in determining optimization choices and constraints.
| | |
|:---:|:----:|
| CPU | [Intel(R) Core(TM) i7-4770 CPU @ 3.40GHz](https://ark.intel.com/content/www/us/en/ark/products/75122/intel-core-i7-4770-processor-8m-cache-up-to-3-90-ghz.html) |
| RAM | 32 GB |
| Cores | 4 |
| Threads | 8 |
| L1d | 32K |
| L1i | 32K | 
| L2 | 256K |
| L3 | 8192K |

TAs/Tutors who helped us: Kunal Dutta, Luke Mujica, Jie Qiu, Cynthia Zhong, Kevin Lafeur, Dayeol Lee