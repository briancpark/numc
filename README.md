# numc
Kaelyn Kim, Brian Park

Here's what I did in project 4:
-

## Task 1
Implemented various functions. There were some conceptual challenges in understanding how slicing works, but turns out they're really represented as linked lists/trees of matrix structs. So all we needed to do was shift memory pointers by offsets. Naive matrix operations were actually pretty straightfoward, and *much* easier compared to the matrix operations we had to implement in RISC-V assembly for project 2. Optimizations were applied later once we learn parallelism!

### `int allocate_matrix(matrix **mat, int rows, int cols)`
The very first function we implemented. In order to understand this and the spec inside and out, we took the time to carefully implement everything before moving on to other function. An important design choice was to use a double pointer array to properly address and call on 2D matrices. It makes sense to do so, and made slicing implementation much more intuitive and less of a hassle. Later we optimized this in Task 4.

### `int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset, int rows, int cols)`
Understanding `allocate_matrix()` allowed us to proceed further to this function. It turns out this was related to slicing. We had to come back later and fix bugs related to pointers. For the longest time, Brian was stuck up on how to shift pointers around and weird bugs would happen. After Brian rigorously debugged in `cgdb`, he found the pointers could be elegantly be shifted with:
```c
(*mat)->data[i] = from->data[i + row_offset] + col_offset;
```
### `void deallocate_matrix(matrix *mat)`
A bit complicated as we had to deal with how to allocate parents and child matrices, but soon realized it was really just linked list/tree data structure. We just have to recursively check on parent/child matrices if they are referenced or need to be `free()`'d.
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
Naive solution was a bit tricky to implement, as we realized that trying to multiply a matrix by itself would scribble data from the original matrix and overwrite it. Thus, we had to allocate another temporary array during matrix operation.
### `int pow_matrix(matrix *result, matrix *mat, int pow)`
Naive solution used just a for loop of `mul_matrix()`.
### `int neg_matrix(matrix *result, matrix *mat)`
Very simple, just multiply by -1 for each entry.
### `int abs_matrix(matrix *result, matrix *mat)`
Also very simple, but encountered a bug where the C standard `abs()` function would return an int. This was simply changed to `fabs()` (for floating point absolute value) and our bug was fixed!

## Task 2
Pretty simple, and straightforward. Skimming through the [`Python3` doc](https://docs.python.org/3.6/distutils/apiref.html) helped us solve the problem. Now our Python can communicate with out C code and vice versa! Woohoo!

## Task 3
For this task, a lot of the functions were very simple after truly understanding the `Python` docs. The `Python` docs were confusing to read at first, but rewarding when we slowly caught up with it and make things work! 

Hardest part were slicing, just due to the sheer complexity and *MANY* different errors we needed to handle. 

We also kept failing `set()` for the most minor and funniest reason. Brian kept improving on slicing functionality, thinking that the autograder's set correctness test robustly tested on it. But turns our it was resolved in OH when Brian realized that `set()` didn't have to be invoked through slicing. It could be invoked through `set(self, i, j, val)` in `Python`. The bug was just a simple error handling between `if (!PyLong_Check(val) || !PyFloat_Check(val))` to `if (!PyLong_Check(val) && !PyFloat_Check(val))` Even though time debugging it was fustrating, at least Brian made sure many slicing errors were handled and properly working. 

### Testing
After the core parts of Task 3 was implemented, we could finally move on to testing our `numc` library and ensure that it works correctly. The Python `unittest` framework can be abused through tactical testing and Brian added fuzz, scaling, and fuzz repetition global parameters to scale up the testing when needed. Parameters were catiously set and tuned as Brian almost crashed an entire Hive server for running large tests. Brian also crashed a server for not realizing there was a memory leak in `deallocate_matrix()` and the server would hit 32GB of RAM and then die.

For efficient TDD workflow, Brian made a simple bash script to compile and run all tests under the executable 
```
./skiddie.sh
```
It will do everything in one line, (load `Python` environment, clean, compile, and run `unittests`) because well... Brian is lazy and a script kiddie.

## Task 4
Here begins the important chunk of the project, performing these operations faster. These operations are *embarassingly parallel*, so let's abuse it! We got the hardware, so let's write the software!
### Simple (Addition, Subtraction, Negation, Absolute)
Began first by trying to improve the performance of add. Once we knew how to accelerate add, the other simple operations could follow the same structure. 

*The buildup to achieve 5X perfomance is explained through our acceleration of the `add_matrix()` function.*

#### The Naive Solution
First, we began with the naive solution, and turns out it performs about the same as `dumbpy` as expected. 

```c
for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat1->cols; j++) {
        result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];  
    }
}
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

This causes memory to be fragmented. Here lies the memory addresses for a sample 4x4 matrix. This was calculated and debugged through `gdb`. We see that memory is fragmented in randomized areas in memory by rows.
| Row/Col Index |     0    |     1    |     2    |     3    |
|--------------:|:--------:|:--------:|:--------:|:--------:|
|             0 | 0xe90d20 | 0xe90d28 | 0xe90d30 | 0xe90d38 |
|             1 | 0xa8e9a0 | 0xa8e9a0 | 0xa8e9a0 | 0xa8e9a0 |
|             2 | 0xe8ddb0 | 0xe8ddb8 | 0xe8ddc0 | 0xe8ddc8 |
|             3 | 0xea0490 | 0xea0498 | 0xea04a0 | 0xea04a8 |

As you can observe, a simple `calloc()` is performed for the whole `mat->data`. Then `rows` are separately malloced to populate it with where each row starts in the row-major 1D array. This is the key to making matrix data still callable through `mat->data[i][j]`. This also improved caching and causes less misses since the memory is now contiguous and improved on spatial locality.

```c
double *arr = calloc(rows * cols, sizeof(double));
(*mat)->data = malloc(rows * sizeof(double*));
for (int i = 0; i < rows; i++) {
    (*mat)->data[i] = &(arr[i * cols]);
}
```

And here is how the memory addresses look like after revising it to be 1D row major order for a 4x4 matrix, again debugged through `gdb`.

| Row/Col Index |     0    |     1    |     2    |     3    |
|--------------:|:--------:|:--------:|:--------:|:--------:|
|             0 | 0xe65940 | 0xe65948 | 0xe65950 | 0xe65958 |
|             1 | 0xe65960 | 0xe65968 | 0xe65970 | 0xe65978 |
|             2 | 0xe65980 | 0xe65988 | 0xe65990 | 0xe65998 |
|             3 | 0xe659a0 | 0xe659a8 | 0xe659b0 | 0xe659b8 |


#### Speed it up with SIMD
Time to start speeding it up even more! We applied SIMD with [Intel Intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2&expand=2184,2183,2177,3980,5999,6006,3594,5392,956). [Linked here](https://ark.intel.com/content/www/us/en/ark/products/75122/intel-core-i7-4770-processor-8m-cache-up-to-3-90-ghz.html) is the specs of the [Hive machines](https://www.ocf.berkeley.edu/~hkn/hivemind/). Vectorization provided near 3X speedup compared to `dumbpy`. We used the hints provided in the spec on which intrinsics should be used. We mainly used 256 bit vector operations as those were the maximum data provided by the Hive's Intel i7 4th generation processors. We also unrolled the loops for a minimal speed up, although we are not sure if that actually makes a difference. We weren't sure if the compiler actually automatically unrolls it, but did it to just stay safe!

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
//Tail Case
for (int i = mat1->rows * mat1->cols / 16 * 16; i < mat1->rows * mat1->cols; i++) {
    *(res_pointer + i) = *(mat1_pointer + i) + *(mat2_pointer + i); 
}
```

#### I hope the rope is... Multithreaded
Faster! We applied OpenMP, with a simple `pragma omp parallel for`. Was it really that easy though? No, as you saw in the previous iteration with SIMD, we had to stuff all the operations in one line. We did this to prevent any race conditions or false sharing that would happen with parallelization. Even though the Hive's machine have 8 threads, they are hyperthreaded, and the computer architecture is really just 4 cores. Hyperthreading makes it so that the 2 threads in a core would compete for data, so we catiously chose to make it run on 4 threads instead. This gives us a total speedup of 5X compared to `dumbpy`, sometimes 5.1X if lucky!

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
//Tail Case
for (int i = mat1->rows * mat1->cols / 16 * 16; i < mat1->rows * mat1->cols; i++) {
    *(res_pointer + i) = *(mat1_pointer + i) + *(mat2_pointer + i); 
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
}
```

#### Can We Do Even Better? (Conclusion)
We could certainly optimize it a bit more efficiently by carefully thinking about caches and virtual memory. But this is the best we could come with in terms of performance. 


### Multiply
This was mainly the hardest part of the project. Although we thought we have mastered matrix multiplication by doing it in RISC-V for project 2, it is even harder trying to parallelize it. There are things you need to consider such as how memory is meticuously handled. We will explain our troubles and difficulty spent debugging through the buildup of how matrix multiplication was sped up. Brian did a *LOT* of research on DGEMM papers. DGEMM stands for **D**ouble-precision, **GE**neral **M**atrix-**M**atrix multiplication. Resources include [Patterson and Hennesy's Computer Organization and Design](https://www.amazon.com/Computer-Organization-Design-RISC-V-Architecture/dp/0128122757), [Nicholas Weaver's 61C - Lecture 18 Spring 2019](https://www.youtube.com/watch?v=ibzkJAkn2_o) [slides](https://inst.eecs.berkeley.edu/~cs61c/sp19/lectures/lec18.pdf), [What Every Programmer Should Know About Memory by Ulrich Drepper](https://akkadia.org/drepper/cpumemory.pdf), and [Matrix Multiplication using SIMD](https://www.youtube.com/watch?v=3rU6BX7w8Tk&list=PLKT8ER2pEV3umVSMwd06LY_eSIX-DnU6A&index=1).

#### The Naive Solution
Very simple. After project 2, this was really not a challenge at all.

Just note that a copy of the matrix is made with `*data`. This is necessary in order to prevent memory overwrite when doing operations like `pow_matrix()`, which you will see later that it writes over `mat1->data` and `mat2->data`. 

*(e.g.) Something in `python3` like this snippet of code can make things problematic:*
```python3
nc_mat1 = nc_mat1 * nc_mat2
```

Thus, here is the implementation of the naive version.
```c
double *data = (double*) calloc(mat1->rows * mat2->cols, sizeof(double));

for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat2->cols; j++) {
        for (int k = 0; k < mat2->rows; k++) {
            *(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
        }
    }
}
```

Before we move on, we must first observe why matrix multiplication is such a complex operation to optimize. We see that we keep hitting strides of the matrix b at the innermost for loop, we have to mitigate this or else, we miss a lot of cache hits.

![matmul](https://www.mymathtables.com/calculator/matrix/3x3-matrix-formula.png)

#### SIMD
Again, Intel Intrinsics saves the day with subword parallelism. Fortunately, Patterson and Henessy paints the picture very elegantly in their textbook, as their newest edition includes a buildup of how to improve DGEMM performance with SIMD, cache blocking, and multithreaded parallelism. Unfortunately, the Patterson and Hennessy implementation does not work out of the box, because they only did it for square matrices with dimension of `2^n`, so a tail case needed to be implemented as well. Here is how that looks like with a few more optimizations like loop unrolling as well:

```c
for (int i = 0; i < mat1->rows; i++) { 
    for (int j = 0; j < mat2->cols / 16 * 16; j += 16) {
        __m256d c0 = _mm256_loadu_pd(data + (i * mat2->cols) + j);
        __m256d c1 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 4);
        __m256d c2 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 8);
        __m256d c3 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 12);

        for (int k = 0; k < mat2->rows; k++) {                                
            c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j), c0);
            c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j + 4), c1);
            c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j + 8), c2);
            c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j + 12), c3);                         
        }
        _mm256_storeu_pd(data + (i * mat2->cols) + j, c0); 
        _mm256_storeu_pd(data + (i * mat2->cols) + j + 4, c1); 
        _mm256_storeu_pd(data + (i * mat2->cols) + j + 8, c2); 
        _mm256_storeu_pd(data + (i * mat2->cols) + j + 12, c3); 
    }
}

for (int i = 0; i < mat1->rows; i++) {
    for (int j = mat2->cols / 16 * 16; j < mat2->cols; j++) {
        for (int k = 0; k < mat2->rows; k++) {
            *(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
        }
    }
}
```

#### Cache Blocking
We can do even better with the naive implementation by introducing cache blocking. Looking at the 4th generation i7 architecture, it has a 32K instruction and data cache. So we used that to our advantage and cache blocked the operations for better cache performance, causing less misses, more hits. This was inspired heavily from the Patterson and Hennessy implemenation, but other research papers call this cache tiling. This was the first improvement we started out with, and we acheived a near 7X speedup alone with the naive implementation:

```c
int blocksize = 32;
for (int i_blocked = 0; i_blocked < mat1->rows; i_blocked += blocksize) {
    for (int j_blocked = 0; j_blocked < mat2->cols; j_blocked += blocksize) {
        for (int k_blocked = 0; k_blocked < mat2->rows; k_blocked += blocksize) {
            for (int i = i_blocked; i < (i_blocked + blocksize) && i < mat1->rows; i++) {
                for (int j = j_blocked; j < j_blocked + blocksize && j < mat2->cols; j++) {        
                    for (int k = k_blocked; k < k_blocked + blocksize && k < mat2->rows; k++) {
                        *(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
                    }
                }
            }
        }
    }
}
```

#### Multithreaded
We have 6 `for` loops with all these optimizations. We simply put `#pragma omp parallel for num_threads(4)` on the outer most `for` loop. And that helped us achieve near 120X perfomance.

```c
#pragma omp parallel for num_threads(4)
for (int i_blocked = 0; i_blocked < mat1->rows; i_blocked += blocksize) {
    for (int j_blocked = 0; j_blocked < mat2->cols; j_blocked += blocksize) {
        for (int k_blocked = 0; k_blocked < mat2->rows; k_blocked += blocksize) {
            for (int i = i_blocked; i < (i_blocked + blocksize) && i < mat1->rows; i++) { 
                for (int j = j_blocked; j < j_blocked + blocksize && j < mat2->cols / 16 * 16; j += 16) {
                    __m256d c0 = _mm256_loadu_pd(data + (i * mat2->cols) + j);
                    __m256d c1 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 4);
                    __m256d c2 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 8);
                    __m256d c3 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 12);

                    for (int k = k_blocked; k < k_blocked + blocksize && k < mat2->rows; k++) {                                
                        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j), c0);
                        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j + 4), c1);
                        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j + 8), c2);
                        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(mat1->data[i] + k), _mm256_loadu_pd(mat2->data[k] + j + 12), c3);                         
                    }
                    _mm256_storeu_pd(data + (i * mat2->cols) + j, c0); 
                    _mm256_storeu_pd(data + (i * mat2->cols) + j + 4, c1); 
                    _mm256_storeu_pd(data + (i * mat2->cols) + j + 8, c2); 
                    _mm256_storeu_pd(data + (i * mat2->cols) + j + 12, c3); 
                }
            }
        }
    }
}

#pragma omp parallel for num_threads(4)
for (int i = 0; i < mat1->rows; i++) {
    for (int j = mat2->cols / 16 * 16; j < mat2->cols; j++) {
        for (int k = 0; k < mat2->rows; k++) {
            *(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
        }
    }
}
```

#### Failed Idea (Matrix Transpose)
This was one of our first drafts with matrix multiplication improvements, and we thought transforming the matrix would work, as the dot products can be efficiently organized and multiplied. But this only gave us a 45X speedup. It was probably due to a bottleneck of trying to transpose and copy the matrix down, not pulling down the amortized runtime. Also there are too many `_mm256_storeu_pd` going on, which can hurt performance. So eventually we had to scrap this early on. Fortunately, we started the project early and were able to come up with the 120X solution shown above.

```c
double *b_t_rows = (double*) calloc(mat2->rows * mat2->cols, sizeof(double));
if (b_t_rows== NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
    return -1;
}

double **b_transpose = (double**) malloc(mat2->cols * sizeof(double*));        
if (b_transpose == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
    return -1;
}

#pragma omp parallel for num_threads(4)
for (int i = 0; i < mat2->cols; i++) {
    b_transpose[i] = &(b_t_rows[i * mat2->rows]);
}

#pragma omp parallel for num_threads(4)
for (int i = 0; i < mat2->cols; i++) {
    for (int j = 0; j < mat2->rows; j++) {
        b_transpose[i][j] = mat2->data[j][i];
    }
}

int blocksize = 64;

if (mat1->rows < blocksize || mat1->cols < blocksize || mat2->rows < blocksize || mat2->cols < blocksize || mat1->parent != NULL || mat2->parent != NULL) {
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat2->cols; j++) {
            for (int k = 0; k < mat2->rows; k++) {
                *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
            }
        }
    }

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < mat1->rows * mat2->cols; i++) {
        *(result->data[0] + i) = *(data + i);                
    }
}

#pragma omp parallel for num_threads(4)
for (int j_blocked = 0; j_blocked < mat2->cols; j_blocked += blocksize) {
    for (int k_blocked = 0; k_blocked < mat2->rows; k_blocked += blocksize) {
        for (int i_blocked = 0; i_blocked < mat1->rows; i_blocked += blocksize) {        
            for (int k = k_blocked; k < k_blocked + blocksize && k < mat2->rows / 16 * 16; k += 16) {                                
                for (int j = j_blocked; j < j_blocked + blocksize && j < mat2->cols; j++) {
                    for (int i = i_blocked; i < (i_blocked + blocksize) && i < mat1->rows; i++) {
                        double sum1[4];
                        _mm256_storeu_pd(sum1, _mm256_mul_pd(_mm256_loadu_pd(mat1->data[i] + k), _mm256_loadu_pd(b_transpose[j] + k)));
                        *(data + (i * mat2->cols) + j) += (sum1[0] + sum1[1] + sum1[2] + sum1[3]);
                        double sum2[4];
                        _mm256_storeu_pd(sum2, _mm256_mul_pd(_mm256_loadu_pd(mat1->data[i] + k + 4), _mm256_loadu_pd(b_transpose[j] + k + 4)));
                        *(data + (i * mat2->cols) + j) += (sum2[0] + sum2[1] + sum2[2] + sum2[3]);
                        double sum3[4];
                        _mm256_storeu_pd(sum3, _mm256_mul_pd(_mm256_loadu_pd(mat1->data[i] + k + 8), _mm256_loadu_pd(b_transpose[j] + k + 8)));
                        *(data + (i * mat2->cols) + j) += (sum3[0] + sum3[1] + sum3[2] + sum3[3]);
                        double sum4[4];
                        _mm256_storeu_pd(sum4, _mm256_mul_pd(_mm256_loadu_pd(mat1->data[i] + k + 12), _mm256_loadu_pd(b_transpose[j] + k + 12)));
                        *(data + (i * mat2->cols) + j) += (sum4[0] + sum4[1] + sum4[2] + sum4[3]);
                    }
                }
            }
        }
    }
}

#pragma omp parallel for num_threads(4)
for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat2->cols; j++) {
        for (int k = mat2->rows / 16 * 16; k < mat2->rows; k++) {
            *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
        }
    }
}
free(b_transpose[0]);
free(b_transpose);
```

#### Infeasible Ideas (Strassen's)
We debated over this one heavily. Having taken CS 170, we thought this would be a very nice divide and conquer method, being easily parallizable and cut down on runtime. Although Strassen's performs O(n^(2.81)) compared to O(n^3), there was an issue with how Strassens work. Strassens performs poorly on smaller matrices. Also, Strassen's requires matrix dimensions to be a power of 2 with it being square. This is possible to do by zero padding matrices, but we also lose performance if the matrix is not relatively square or we're doing matrix-vector multiplication. We could be working with very sparse matrices if the dimensions are unaligned. `allocate_matrix()` would need to be tuned to zero pad matrices, potentially messing with the functionality of other matrix operations. Cache blocking and SIMD would be operated on data with `0.0` if it's zero padded, so it sounds like a bad algorithm to implement. We'd be wasting a lot of computation on zero vectors on sparse matrices. Thus, the improvement from O(n^3) to O(n^(2.81)) was not worth it due to the input limitations of Strassen's.

#### Can We Do Even Better (Conclusion)
Of course, DGEMM is sill being researched today. This may be the best we can do in terms of hardware and the limitations of the Hive's 4th generation i7. There are certainly other types of DGEMM research going on related to other types of hardware such as GPUs, TPUs, and even Apple's new A12 with Neural Engine, all with high DGEMM performance to compute neural networks and other matrix operations. It is able to [make our iPhones much faster and powerful](https://analyticsindiamag.com/apple-a14-bionic-machine-learning-chip-processor/), [beat a professional player in Go](https://www.theverge.com/circuitbreaker/2016/5/19/11716818/google-alphago-hardware-asic-chip-tensor-processor-unit-machine-learning), and [even play 8K HDR gaming!](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/).

Even NVIDIA has their own type of Intrinsic-like parallel programming platform,  [CUDA](https://developer.nvidia.com/cuda-toolkit). We've seen it outperform in Deep Learning applications and gaming, and this might be a fun project to learn/do over break now that we have done it successfully in Intel CPU architecture!

### Power
Used a simple divide and conquer method noted [here](https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/). The way matrices are powered in this once cuts down coputation by an order of O(log(n)) relative to matrix multiplication. The actual runtime would be O(n^3log(n)), but because parallelism was applied, it's a bit weird to do formal runtime analyis.

#### Matrix Exponentation
Here is how it's done! Basically we and save computation on an order of log(n) by repeatedly squaring A. 

*e.g. A^5 = ((A^2)^2)A*

This in total gives us over 2000X performance!

#### Matrix Decomposition (Draft)
We can definitely do better. The total runtime of O(n^3log(n)) can be done better, but in tricky cases with high manipulation. We can try to exploit linear algebra by using [spectral decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix). Basically, a matrix A can be decomposed into A = VDV^-1. Although the computation to get the eigenvalues and eigenvector to compose V, D, and V^-1, it should not even matter in terms of amortized cost and perfomance. Because now you can use regular matrix multiplication to compute A^n. Why? 

A^n = V * D * V^-1 * V * D * V^-1 * ... * V * D * V^-1 (decomposition is multiplied n times)
A^n = V * D^n * V^-1 (exploit the fact that V^-1 * V = I)

Of course, there are certain cases to consider, cannot be defective and must be diagonalizable, so might be infeasible???.......

**This idea is still in progress... write more here**


### About the Hive CPUs
Will be useful in determining optimization choices and constraints.
| Specification |  |
|:---:|:----:|
| CPU | [Intel(R) Core(TM) i7-4770 CPU @ 3.40GHz](https://ark.intel.com/content/www/us/en/ark/products/75122/intel-core-i7-4770-processor-8m-cache-up-to-3-90-ghz.html) |
| RAM | 32 GB |
| Cores | 4 |
| Threads | 8 |
| L1d | 32K |
| L1i | 32K | 
| L2 | 256K |
| L3 | 8192K |

## Acknowledgements
TAs/Tutors who helped us: Kunal Dutta, Luke Mujica, Jie Qiu, Cynthia Zhong, Kevin Lafeur, Dayeol Lee

##
*“People who are really serious about software should make their own hardware.”* --Alan Kay