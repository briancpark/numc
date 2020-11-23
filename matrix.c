#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    //How to allocate continguous memory was referenced here: https://www.dimlucas.com/index.php/2017/02/18/the-proper-way-to-dynamically-create-a-two-dimensional-array-in-c/
    if (rows < 1 || cols < 1) {
        PyErr_SetString(PyExc_TypeError, "Nonpositive dimensions!");
        return -1;
    }

    (*mat) = (matrix*) malloc(sizeof(struct matrix));

    if ((*mat) == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
        return -1;
    }

    (*mat)->rows = rows;
    (*mat)->cols = cols;

    if (rows == 1 || cols == 1) {
        (*mat)->is_1d = 1;
    } else {
        (*mat)->is_1d = 0;
    }

    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;

    double *arr = calloc(rows * cols, sizeof(double));
    if (arr == NULL) {
        free((*mat));
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
        return -1;
    }

    (*mat)->data = malloc(rows * sizeof(double*));
    if ((*mat)->data == NULL) {
        free(arr);
        free((*mat));
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
        return -1;
    }

    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        (*mat)->data[i] = &(arr[i * cols]);
    }

    /*
    if ((*mat)->data == NULL) {
        free((*mat));
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
        return -1;
    }

    for (int i = 0; i < rows; i++) {
        (*mat)->data[i] = (double *) calloc(cols, sizeof(double));
        if ((*mat)->data[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free((*mat)->data[j]);
            }
            free((*mat)->data);
            free((*mat));
            PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
            return -1;
        }
    }
    */

    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows < 0 || cols < 0) {
        PyErr_SetString(PyExc_TypeError, "Negative index");
        return -1;
    }

    (*mat) = (matrix*) malloc(sizeof(struct matrix));

    if ((*mat) == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed.");
        return -1;
    }

    (*mat)->rows = rows;
    (*mat)->cols = cols;

    if (rows == 1 || cols == 1) {
        (*mat)->is_1d = 1;
    } else {
        (*mat)->is_1d = 0;
    }

    from->ref_cnt++;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = from;

    (*mat)->data = (double**) malloc(sizeof(double*) * rows);
    if ((*mat)->data == NULL) {
        free((*mat));
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        (*mat)->data[i] = from->data[i + row_offset] + col_offset;

        if ((*mat)->data[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free((*mat)->data[j]);
            }
            free((*mat)->data);
            free((*mat));
            PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
            return -1;
        }
    }

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // The process of deallocating is much similar to deallocating/freeing a linked list in C
    
    if (mat == NULL) {
        return;
    } else if (mat->parent == NULL && mat->ref_cnt <= 1) {
        free(mat->data[0]);
        free(mat->data);
        free(mat);
    } else if (mat->parent != NULL && mat->parent->ref_cnt <= 1) {
        deallocate_matrix(mat->parent);
        free(mat);
    } else {
        mat->ref_cnt--;
    }
}    

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    int rows = mat->rows;
    int cols = mat->cols;
    /*
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat->data[i][j] = val;
        }
    }
    */
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < rows * cols; i++) {
        *(mat->data[0] + i) = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((result->rows == mat1->rows && mat1->rows == mat2->rows) && 
        (result->cols == mat1->cols && mat1->cols == mat2->cols)) {

        /*
        //Naive solution
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
            }
        }
        return 0;
        */
         
        //Unrolled naive
        /*
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols / 4 * 4 ; j += 4) {
                result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
                result->data[i][j + 1] = mat1->data[i][j + 1] + mat2->data[i][j + 1];
                result->data[i][j + 2] = mat1->data[i][j + 2] + mat2->data[i][j + 2];
                result->data[i][j + 3] = mat1->data[i][j + 3] + mat2->data[i][j + 3];
            }

            for(int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
                result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
            }
        }
        */
 
        /* SIMD Acceleration
         * -Floating point should follow IEEE 754 (64 bit) 8 bytes standard
         * -Thus 4 doubles will fit into 256bit vector
         * -Intuitively, we should load vectors, add vectors to a new resultant vector, 
         *  and then store the resultant vector back into memory (*result)
         */
        
        /*
        __m256d mat1_vector;
        __m256d mat2_vector;
        __m256d sum_vector;

        for (int i = 0; i < mat1->rows; i++) {
            double *pointer1 = mat1->data[i];
            double *pointer2 = mat2->data[i];
            double *result_pointer = result->data[i];
            for (int j = 0; j < mat2->cols / 4 * 4; j += 4) {   
                mat1_vector = _mm256_loadu_pd(pointer1);
                mat2_vector = _mm256_loadu_pd(pointer2);

                sum_vector = _mm256_add_pd(mat1_vector, mat2_vector);
                
                _mm256_storeu_pd(result_pointer, sum_vector);
                pointer1 += 4;
                pointer2 += 4;
                result_pointer += 4;
            }

            for(int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
                result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
            }
        }
        */


        //Vanilla MIMD with no race condition
        /*
        double *pointer1 = mat1->data[0];
        double *pointer2 = mat2->data[0];
        double *result_pointer = result->data[0];

        omp_set_num_threads(8);
        #pragma omp parallel for
        for(int i = 0; i < mat2->cols * mat2->rows; i++) {
            *(result_pointer + i) = *(pointer1 + i) + *(pointer2 + i);
        }
        */

        // SIMD/MIMD FULL SPEED
        /*
        if (mat1->rows < 150 || mat1->cols < 150) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat2->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
                }
            }
            return 0;
        }
            
        #pragma omp parallel for num_threads(8) collapse(2)
        for (int i = 0; i < mat2->rows; i++) {        
            for (int j = 0; j < mat2->cols / 16 * 16; j += 16) { 
                _mm256_storeu_pd(result->data[i] + j, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j), _mm256_loadu_pd(mat2->data[i] + j)));
                _mm256_storeu_pd(result->data[i] + j + 4, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j + 4), _mm256_loadu_pd(mat2->data[i] + j + 4)));
                _mm256_storeu_pd(result->data[i] + j + 8, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j + 8), _mm256_loadu_pd(mat2->data[i] + j + 8)));
                _mm256_storeu_pd(result->data[i] + j + 12, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j + 12), _mm256_loadu_pd(mat2->data[i] + j + 12)));
            }
            
        }

        //Tail Case
        #pragma omp parallel for num_threads(8) collapse(2)
        for (int i = 0; i < mat2->rows; i++) {  
            for(int j = mat2->cols / 16 * 16; j < mat2->cols; j++) {
                *(result->data[i] + j) = *(mat1->data[i] + j) + *(mat2->data[i] + j);
            }     
        }
        return 0;
        */
        /*
        if (mat1->rows < 16 || mat1->cols < 16) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat1->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];  
                }
            }
            return 0;
        }

        if (mat1->parent != NULL || mat2->parent != NULL) {
            #pragma omp parallel for num_threads(8) collapse(2)
            for (int i = 0; i < mat2->rows; i++) {        
                for (int j = 0; j < mat2->cols / 16 * 16; j += 16) { 
                    _mm256_storeu_pd(result->data[i] + j, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j), _mm256_loadu_pd(mat2->data[i] + j)));
                    _mm256_storeu_pd(result->data[i] + j + 4, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j + 4), _mm256_loadu_pd(mat2->data[i] + j + 4)));
                    _mm256_storeu_pd(result->data[i] + j + 8, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j + 8), _mm256_loadu_pd(mat2->data[i] + j + 8)));
                    _mm256_storeu_pd(result->data[i] + j + 12, _mm256_add_pd(_mm256_loadu_pd(mat1->data[i] + j + 12), _mm256_loadu_pd(mat2->data[i] + j + 12)));
                }
                
            }

            //Tail Case
            #pragma omp parallel for num_threads(8) collapse(2)
            for (int i = 0; i < mat2->rows; i++) {  
                for(int j = mat2->cols / 16 * 16; j < mat2->cols; j++) {
                    *(result->data[i] + j) = *(mat1->data[i] + j) + *(mat2->data[i] + j);
                }     
            }
            return 0;
        }

        if (mat1->rows < 256 || mat1->cols < 256) {
            for (int i = 0; i < mat1->rows * mat1->cols / 4 * 4; i += 4) {
                result->data[i / mat1->cols][i % mat1->cols] = mat1->data[i / mat1->cols][i % mat1->cols] + mat2->data[i / mat2->cols][i % mat2->cols];  
                result->data[(i + 1) / mat1->cols][(i + 1) % mat1->cols] = mat1->data[(i + 1) / mat1->cols][(i + 1) % mat1->cols] + mat2->data[(i + 1) / mat2->cols][(i + 1) % mat2->cols];  
                result->data[(i + 2) / mat1->cols][(i + 2) % mat1->cols] = mat1->data[(i + 2) / mat1->cols][(i + 2) % mat1->cols] + mat2->data[(i + 2) / mat2->cols][(i + 2) % mat2->cols];  
                result->data[(i + 3) / mat1->cols][(i + 3) % mat1->cols] = mat1->data[(i + 3) / mat1->cols][(i + 3) % mat1->cols] + mat2->data[(i + 3) / mat2->cols][(i + 3) % mat2->cols];  
            }
            
            for (int i = mat1->rows * mat1->cols / 4 * 4; i < mat1->rows * mat1->cols; i++) {
                result->data[i / mat1->cols][i % mat1->cols] = mat1->data[i / mat1->cols][i % mat1->cols] + mat2->data[i / mat2->cols][i % mat2->cols];  
            }
            return 0;
        }

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat2->rows * mat2->cols / 16 * 16; i += 16) {        
            _mm256_storeu_pd(result->data[i / mat2->cols] + (i % mat2->cols), _mm256_add_pd(_mm256_loadu_pd(mat1->data[i / mat2->cols] + (i % mat2->cols)), _mm256_loadu_pd(mat2->data[i / mat2->cols] + (i % mat2->cols))));
            _mm256_storeu_pd(result->data[(i + 4) / mat2->cols] + ((i + 4) % mat2->cols), _mm256_add_pd(_mm256_loadu_pd(mat1->data[(i + 4) / mat2->cols] + ((i + 4) % mat2->cols)), _mm256_loadu_pd(mat2->data[(i + 4) / mat2->cols] + ((i + 4) % mat2->cols))));
            _mm256_storeu_pd(result->data[(i + 8) / mat2->cols] + ((i + 8) % mat2->cols), _mm256_add_pd(_mm256_loadu_pd(mat1->data[(i + 8) / mat2->cols] + ((i + 8) % mat2->cols)), _mm256_loadu_pd(mat2->data[(i + 8) / mat2->cols] + ((i + 8) % mat2->cols))));
            _mm256_storeu_pd(result->data[(i + 12) / mat2->cols] + ((i + 12) % mat2->cols), _mm256_add_pd(_mm256_loadu_pd(mat1->data[(i + 12) / mat2->cols] + ((i + 12) % mat2->cols)), _mm256_loadu_pd(mat2->data[(i + 12) / mat2->cols] + ((i + 12) % mat2->cols))));        
        }

        //Tail Case
        #pragma omp parallel for
        for (int i = mat2->rows * mat2->cols / 16 * 16; i < mat2->rows * mat2->cols; i++) {  
            result->data[i / mat1->cols][i % mat1->cols] = mat1->data[i / mat1->cols][i % mat1->cols] + mat2->data[i / mat2->cols][i % mat2->cols];  
        }
        return 0;
        */

        
        if (mat1->rows < 16 || mat1->cols < 16 || mat1->parent != NULL || mat2->parent != NULL) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat1->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];  
                }
            }
            return 0;
        }

        double *res_pointer = result->data[0];
        double *mat1_pointer = mat1->data[0];
        double *mat2_pointer = mat2->data[0];

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat1->rows * mat1->cols / 16 * 16; i += 16) {        
            _mm256_storeu_pd(res_pointer + i, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i), _mm256_loadu_pd(mat2_pointer + i)));
            _mm256_storeu_pd(res_pointer + i + 4, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 4), _mm256_loadu_pd(mat2_pointer + i + 4)));
            _mm256_storeu_pd(res_pointer + i + 8, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 8), _mm256_loadu_pd(mat2_pointer + i + 8)));
            _mm256_storeu_pd(res_pointer + i + 12, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 12), _mm256_loadu_pd(mat2_pointer + i + 12)));
        }      

        //Tail Case

        for (int i = mat1->rows * mat1->cols / 16 * 16; i < mat1->rows * mat1->cols; i++) {
            *(res_pointer + i) = *(mat1_pointer + i) + *(mat2_pointer + i); 
        }
        return 0;


    }
    return -1;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((result->rows == mat1->rows && mat1->rows == mat2->rows) && 
        (result->cols == mat1->cols && mat1->cols == mat2->cols)) {
        /*
        // SIMD/MIMD FULL SPEED
        if (mat1->rows < 16 || mat1->cols < 16) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat1->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];  
                }
            }
            return 0;
        }

        if (mat1->parent != NULL || mat2->parent != NULL) {
            #pragma omp parallel for num_threads(8) collapse(2)
            for (int i = 0; i < mat2->rows; i++) {        
                for (int j = 0; j < mat2->cols / 16 * 16; j += 16) { 
                    _mm256_storeu_pd(result->data[i] + j, _mm256_sub_pd(_mm256_loadu_pd(mat1->data[i] + j), _mm256_loadu_pd(mat2->data[i] + j)));
                    _mm256_storeu_pd(result->data[i] + j + 4, _mm256_sub_pd(_mm256_loadu_pd(mat1->data[i] + j + 4), _mm256_loadu_pd(mat2->data[i] + j + 4)));
                    _mm256_storeu_pd(result->data[i] + j + 8, _mm256_sub_pd(_mm256_loadu_pd(mat1->data[i] + j + 8), _mm256_loadu_pd(mat2->data[i] + j + 8)));
                    _mm256_storeu_pd(result->data[i] + j + 12, _mm256_sub_pd(_mm256_loadu_pd(mat1->data[i] + j + 12), _mm256_loadu_pd(mat2->data[i] + j + 12)));
                }
                
            }

            //Tail Case
            #pragma omp parallel for num_threads(8) collapse(2)
            for (int i = 0; i < mat2->rows; i++) {  
                for(int j = mat2->cols / 16 * 16; j < mat2->cols; j++) {
                    *(result->data[i] + j) = *(mat1->data[i] + j) - *(mat2->data[i] + j);
                }     
            }
            return 0;
        }

        if (mat1->rows < 256 || mat1->cols < 256) {
            for (int i = 0; i < mat1->rows * mat1->cols / 4 * 4; i += 4) {
                result->data[i / mat1->cols][i % mat1->cols] = mat1->data[i / mat1->cols][i % mat1->cols] - mat2->data[i / mat2->cols][i % mat2->cols];  
                result->data[(i + 1) / mat1->cols][(i + 1) % mat1->cols] = mat1->data[(i + 1) / mat1->cols][(i + 1) % mat1->cols] - mat2->data[(i + 1) / mat2->cols][(i + 1) % mat2->cols];  
                result->data[(i + 2) / mat1->cols][(i + 2) % mat1->cols] = mat1->data[(i + 2) / mat1->cols][(i + 2) % mat1->cols] - mat2->data[(i + 2) / mat2->cols][(i + 2) % mat2->cols];  
                result->data[(i + 3) / mat1->cols][(i + 3) % mat1->cols] = mat1->data[(i + 3) / mat1->cols][(i + 3) % mat1->cols] - mat2->data[(i + 3) / mat2->cols][(i + 3) % mat2->cols];  
            }
            
            for (int i = mat1->rows * mat1->cols / 4 * 4; i < mat1->rows * mat1->cols; i++) {
                result->data[i / mat1->cols][i % mat1->cols] = mat1->data[i / mat1->cols][i % mat1->cols] - mat2->data[i / mat2->cols][i % mat2->cols];  
            }
            return 0;
        }

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat2->rows * mat2->cols / 16 * 16; i += 16) {        
            _mm256_storeu_pd(result->data[i / mat2->cols] + (i % mat2->cols), _mm256_sub_pd(_mm256_loadu_pd(mat1->data[i / mat2->cols] + (i % mat2->cols)), _mm256_loadu_pd(mat2->data[i / mat2->cols] + (i % mat2->cols))));
            _mm256_storeu_pd(result->data[(i + 4) / mat2->cols] + ((i + 4) % mat2->cols), _mm256_sub_pd(_mm256_loadu_pd(mat1->data[(i + 4) / mat2->cols] + ((i + 4) % mat2->cols)), _mm256_loadu_pd(mat2->data[(i + 4) / mat2->cols] + ((i + 4) % mat2->cols))));
            _mm256_storeu_pd(result->data[(i + 8) / mat2->cols] + ((i + 8) % mat2->cols), _mm256_sub_pd(_mm256_loadu_pd(mat1->data[(i + 8) / mat2->cols] + ((i + 8) % mat2->cols)), _mm256_loadu_pd(mat2->data[(i + 8) / mat2->cols] + ((i + 8) % mat2->cols))));
            _mm256_storeu_pd(result->data[(i + 12) / mat2->cols] + ((i + 12) % mat2->cols), _mm256_sub_pd(_mm256_loadu_pd(mat1->data[(i + 12) / mat2->cols] + ((i + 12) % mat2->cols)), _mm256_loadu_pd(mat2->data[(i + 12) / mat2->cols] + ((i + 12) % mat2->cols))));        
        }

        //Tail Case
        #pragma omp parallel for
        for (int i = mat2->rows * mat2->cols / 16 * 16; i < mat2->rows * mat2->cols; i++) {  
            result->data[i / mat1->cols][i % mat1->cols] = mat1->data[i / mat1->cols][i % mat1->cols] - mat2->data[i / mat2->cols][i % mat2->cols];  
        }
        return 0;
        */
        if (mat1->rows < 16 || mat1->cols < 16 || mat1->parent != NULL || mat2->parent != NULL) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat1->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];  
                }
            }
            return 0;
        }

        double *res_pointer = result->data[0];
        double *mat1_pointer = mat1->data[0];
        double *mat2_pointer = mat2->data[0];

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat1->rows * mat1->cols / 16 * 16; i += 16) {        
            _mm256_storeu_pd(res_pointer + i, _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i), _mm256_loadu_pd(mat2_pointer + i)));
            _mm256_storeu_pd(res_pointer + i + 4, _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i + 4), _mm256_loadu_pd(mat2_pointer + i + 4)));
            _mm256_storeu_pd(res_pointer + i + 8, _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i + 8), _mm256_loadu_pd(mat2_pointer + i + 8)));
            _mm256_storeu_pd(res_pointer + i + 12, _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i + 12), _mm256_loadu_pd(mat2_pointer + i + 12)));
        }

        //Tail Case

        for (int i = mat1->rows * mat1->cols / 16 * 16; i < mat1->rows * mat1->cols; i++) {
            *(res_pointer + i) = *(mat1_pointer + i) - *(mat2_pointer + i); 
        }
        return 0;

    }
    return -1;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((result->rows == mat1->rows) && (result->cols == mat2->cols) && (mat1->cols == mat2->rows)) { 
        
        double *data = (double*) calloc(mat1->rows * mat2->cols, sizeof(double));
        if (data == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
            return 1;
        }
        
 

        /*
        //Naive cached improved solution
        for (int j = 0; j < mat2->cols; j++) {
            for (int k = 0; k < mat2->rows; k++) {
                for (int i = 0; i < mat1->rows; i++) {
                    *(data + (i * mat2->cols) + j) += get(mat1, i, k) * get(mat2, k, j);
                }
            }
        }

        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                set(result, i, j, *(data + (i * mat2->cols) + j));
            }
        }
        */

        //Naive cache blocking, include later as a small matrix check
        /*
        if (mat1->rows < 32 || mat1->cols < 32 || mat2->rows < 32 || mat2->cols < 32) {
            for (int j = 0; j < mat2->cols; j++) {
                for (int k = 0; k < mat2->rows; k++) {
                    for (int i = 0; i < mat1->rows; i++) {
                        *(data + (i * mat2->cols) + j) += get(mat1, i, k) * get(mat2, k, j);
                    }
                }
            }
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat2->cols; j++) {
                    set(result, i, j, *(data + (i * mat2->cols) + j));
                }
            }
            free(data);
            return 0;
        }
        */

        /*
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

        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                set(result, i, j, *(data + (i * mat2->cols) + j));
            }
        }
        free(data);
        return 0;
        */

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

        for (int i = 0; i < mat2->cols; i++) {
            b_transpose[i] = &(b_t_rows[i * mat2->rows]);
        }
        
        for (int i = 0; i < mat2->cols; i++) {
            for (int j = 0; j < mat2->rows; j++) {
                b_transpose[i][j] = mat2->data[j][i];
            }
        }
        
        //Fix this weird transpose bug
        /*
        for (int i = 0; i < mat1->rows / 4 * 4; i++) {
            for (int j = 0; j < mat2->cols / 4 * 4; j+=4) {
                __m256d c = _mm256_loadu_pd(data + (i * mat2->cols) + j);
                for (int k = 0; k < mat2->rows / 4 * 4; k++) {
                    __m256d b = _mm256_loadu_pd(b_transpose[j] + k);
                    c = _mm256_fmadd_pd(_mm256_loadu_pd(mat1->data[i] + k), b, c);
                }
                _mm256_storeu_pd(data + (i * mat2->cols) + j ,c);
                // *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
            }
        }
        */
        
        /*
        //transposed weird tail
        for (int i = 0; i < mat1->rows / 4 * 4; i++) {
            for (int j = 0; j < mat2->cols / 4 * 4; j++) {
                for (int k = 0; k < mat2->rows / 4 * 4; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }

                for (int k = mat2->rows / 4 * 4; k < mat2->rows; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }
            }
            for (int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
                for (int k = 0; k < mat2->rows; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }
            }
        }

        for (int i = mat1->rows / 4 * 4; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols / 4 * 4; j++) {
                for (int k = 0; k < mat2->rows / 4 * 4; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }

                for (int k = mat2->rows / 4 * 4; k < mat2->rows; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }
            }
            for (int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
                for (int k = 0; k < mat2->rows; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }
            }
        }
        */

        /*
        //matmul with intrinsics
        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                //Simd for strided
                double dotp = 0;
                for (int k = 0; k < mat2->rows / 4 * 4; k += 4) {
                    __m256d a = _mm256_loadu_pd(mat1->data[i] + k);
                    __m256d b = _mm256_loadu_pd(b_transpose[j] + k);
                    double *sum = (double*) malloc(sizeof(double) * 4);
                    _mm256_storeu_pd(sum, _mm256_mul_pd(a, b));
                    for (int x = 0; x < 4; x++) {
                        dotp += sum[x];
                    }
                    free(sum);
                }
                *(data + (i * mat2->cols) + j) += dotp; 
            }
        }
        
        //Tail case
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                for (int k = mat2->rows / 4 * 4; k < mat2->rows; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                }
            }
        }
        */

        
        if (mat1->rows < 32 || mat1->cols < 32 || mat2->rows < 32 || mat2->cols < 32) {
            omp_set_num_threads(4);
            #pragma omp parallel for
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat2->cols; j++) {
                    for (int k = 0; k < mat2->rows; k++) {
                        *(data + (i * mat2->cols) + j) += mat1->data[i][k] * b_transpose[j][k];
                    }
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < mat1->rows * mat2->cols; i++) {
                *(result->data[0] + i) = *(data + i);                
            }
            return 0;
        }
        
        

        
        int blocksize = 32;
        #pragma omp parallel for num_threads(4)
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
        


        #pragma omp parallel for
        for (int i = 0; i < mat1->rows * mat2->cols; i++) {
            *(result->data[0] + i) = *(data + i);                
        }
        return 0;


        /*
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols / 16 * 16; j += 16) {
                
                __m256d c[4];
                for (int x = 0; x < 4; x++) {
                    c[x] = _mm256_loadu_pd(data+i+x*4+j*mat2->cols);
                }

                for (int k = 0; k < mat2->rows; k++) {
                    //(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
                    
                    
                    
                    __m256d b = _mm256_broadcast_sd(mat2->data[0]+k+j*mat2->cols);
                    for (int x = 0; x < 4; x++) {
                        c[x] = _mm256_fmadd_pd(_mm256_loadu_pd(mat1->data[0]+mat1->rows*k+x*4+i), b, c[x]);
                    }
                    
      
                }

                for (int x = 0; x < 4; x++) {
                    _mm256_storeu_pd(data + (j * mat2->cols) + i, c[x]);
                }
            }
        }

        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                set(result, i, j, *(data + (i * mat2->cols) + j));
            }
        }
        */
    

        return 0;
    
    }
    return -1;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    // Divide and conquer algorithm inspired from https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/ 
    
    if (pow < 0) {
        //ValueError if a is not a square matrix or if pow is negative.
        PyErr_SetString(PyExc_ValueError, "Power is negative!");
        return -1;
    } else if (mat->rows != mat->cols) {
        PyErr_SetString(PyExc_TypeError, "Matrix is not square!");
        return -1;
    } else {
        //Make a copy of matrix
        //Temps needed to be made in order to prevent writing over existing matrices.
        matrix *ret = NULL;
        allocate_matrix(&ret, mat->rows, mat->cols);
        
        for (int i = 0; i < result->rows; i++) {
            set(ret, i, i, 1);
        }
         
        matrix *A = NULL;
        allocate_matrix(&A, mat->rows, mat->cols);

        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                set(A, i, j, get(mat, i, j));
            }
        }
        
        while (pow > 0) {
            if (pow & 1) {
                mul_matrix(ret, ret, A);
            } 
            mul_matrix(A, A, A);
            pow = pow / 2;
        }

        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                set(result, i, j, get(ret, i, j));
            }
        }

        deallocate_matrix(ret);
        deallocate_matrix(A);
        
        return 0;
    }
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if ((result->rows == mat->rows) && (result->cols == mat->cols)) {
        if (mat->rows < 16 || mat->cols < 16 || mat->parent != NULL) {
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i][j] = (-1.0) * mat->data[i][j];
                }
            }
            return 0;
        }

        double *res_pointer = result->data[0];
        double *mat_pointer = mat->data[0];

        __m256d zero_vector = _mm256_set1_pd(0.0);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat->rows * mat->cols / 16 * 16; i += 16) { 
            _mm256_storeu_pd(res_pointer + i, _mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i)));
            _mm256_storeu_pd(res_pointer + i + 4, _mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i + 4)));
            _mm256_storeu_pd(res_pointer + i + 8, _mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i + 8)));
            _mm256_storeu_pd(res_pointer + i + 12, _mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i + 12)));
        }   
    
        //Tail Case
        #pragma omp parallel for
        for (int i = mat->rows * mat->cols / 16 * 16; i < mat->rows * mat->cols; i++) {  
            *(res_pointer + i) = (-1.0) * *(mat_pointer + i);  
        }
        return 0;



        /*
        if (mat->rows < 16 || mat->cols < 16) {
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i][j] = (-1.0) * mat->data[i][j];
                }
            }
            return 0;
        }

        if (mat->rows < 256 || mat->cols < 256) {
            omp_set_num_threads(4);
            #pragma omp parallel for
            for (int i = 0; i < mat->rows * mat->cols / 4 * 4; i += 4) {
                result->data[i / mat->cols][i % mat->cols] = (-1.0) * mat->data[i / mat->cols][i % mat->cols];  
                result->data[(i + 1) / mat->cols][(i + 1) % mat->cols] = (-1.0) * mat->data[(i + 1) / mat->cols][(i + 1) % mat->cols];  
                result->data[(i + 2) / mat->cols][(i + 2) % mat->cols] = (-1.0) * mat->data[(i + 2) / mat->cols][(i + 2) % mat->cols];  
                result->data[(i + 3) / mat->cols][(i + 3) % mat->cols] = (-1.0) * mat->data[(i + 3) / mat->cols][(i + 3) % mat->cols];  
            }
            #pragma omp parallel for
            for (int i = mat->rows * mat->cols / 4 * 4; i < mat->rows * mat->cols; i++) {
                result->data[i / mat->cols][i % mat->cols] = (-1.0) * mat->data[i / mat->cols][i % mat->cols];  
            }

            return 0;
        }
        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat->rows * mat->cols / 16 * 16; i += 16) { 
            _mm256_storeu_pd(result->data[i / mat->cols] + (i % mat->cols), _mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[i / mat->cols] + (i % mat->cols))));
            _mm256_storeu_pd(result->data[(i + 4) / mat->cols] + ((i + 4) % mat->cols), _mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[(i + 4) / mat->cols] + ((i + 4) % mat->cols))));
            _mm256_storeu_pd(result->data[(i + 8) / mat->cols] + ((i + 8) % mat->cols), _mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[(i + 8) / mat->cols] + ((i + 8) % mat->cols))));
            _mm256_storeu_pd(result->data[(i + 12) / mat->cols] + ((i + 12) % mat->cols), _mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[(i + 12) / mat->cols] + ((i + 12) % mat->cols))));
        }   
    
        //Tail Case
        #pragma omp parallel for
        for (int i = mat->rows * mat->cols / 16 * 16; i < mat->rows * mat->cols; i++) {  
            result->data[i / mat->cols][i % mat->cols] = (-1.0) * mat->data[i / mat->cols][i % mat->cols];  
        }
        return 0;
        */
    }
    return 1;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if ((result->rows == mat->rows) && (result->cols == mat->cols)) {
        /*
        if (mat->rows < 16 || mat->cols < 16) {
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i][j] = fabs(mat->data[i][j]);
                }
            }
            return 0;
        }

        if (mat->rows < 256 || mat->cols < 256) {
            for (int i = 0; i < mat->rows * mat->cols / 4 * 4; i += 4) {
                result->data[i / mat->cols][i % mat->cols] = fabs(mat->data[i / mat->cols][i % mat->cols]);  
                result->data[(i + 1) / mat->cols][(i + 1) % mat->cols] = fabs(mat->data[(i + 1) / mat->cols][(i + 1) % mat->cols]);  
                result->data[(i + 2) / mat->cols][(i + 2) % mat->cols] = fabs(mat->data[(i + 2) / mat->cols][(i + 2) % mat->cols]);  
                result->data[(i + 3) / mat->cols][(i + 3) % mat->cols] = fabs(mat->data[(i + 3) / mat->cols][(i + 3) % mat->cols]);  
            }
            
            for (int i = mat->rows * mat->cols / 4 * 4; i < mat->rows * mat->cols; i++) {
                result->data[i / mat->cols][i % mat->cols] = fabs(mat->data[i / mat->cols][i % mat->cols]);  
            }
            return 0;
        }

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat->rows * mat->cols / 16 * 16; i += 16) { 
            _mm256_storeu_pd(result->data[i / mat->cols] + (i % mat->cols), _mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[i / mat->cols] + (i % mat->cols))), _mm256_loadu_pd(mat->data[i / mat->cols] + (i % mat->cols))));
            _mm256_storeu_pd(result->data[(i + 4) / mat->cols] + ((i + 4) % mat->cols), _mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[(i + 4) / mat->cols] + ((i + 4) % mat->cols))), _mm256_loadu_pd(mat->data[(i + 4) / mat->cols] + ((i + 4) % mat->cols))));
            _mm256_storeu_pd(result->data[(i + 8) / mat->cols] + ((i + 8) % mat->cols), _mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[(i + 8) / mat->cols] + ((i + 8) % mat->cols))), _mm256_loadu_pd(mat->data[(i + 8) / mat->cols] + ((i + 8) % mat->cols))));
            _mm256_storeu_pd(result->data[(i + 12) / mat->cols] + ((i + 12) % mat->cols), _mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), _mm256_loadu_pd(mat->data[(i + 12) / mat->cols] + ((i + 12) % mat->cols))), _mm256_loadu_pd(mat->data[(i + 12) / mat->cols] + ((i + 12) % mat->cols))));
        }
        
        //Tail Case
        #pragma omp parallel for
        for (int i = mat->rows * mat->cols / 16 * 16; i < mat->rows * mat->cols; i++) {  
            result->data[i / mat->cols][i % mat->cols] = fabs(mat->data[i / mat->cols][i % mat->cols]);
        }
        return 0;
        */

        if (mat->rows < 16 || mat->cols < 16 || mat->parent != NULL) {
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i][j] = fabs(mat->data[i][j]);
                }
            }
            return 0;
        }
        

        __m256d zero_vector = _mm256_set1_pd(0.0);
        double *res_pointer = result->data[0];
        double *mat_pointer = mat->data[0];
        
        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < mat->rows * mat->cols / 16 * 16; i += 16) { 
            _mm256_storeu_pd(res_pointer + i, _mm256_max_pd(_mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i)), _mm256_loadu_pd(mat_pointer + i)));
            _mm256_storeu_pd(res_pointer + i + 4, _mm256_max_pd(_mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i + 4)), _mm256_loadu_pd(mat_pointer + i + 4)));
            _mm256_storeu_pd(res_pointer + i + 8, _mm256_max_pd(_mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i + 8)), _mm256_loadu_pd(mat_pointer + i + 8)));
            _mm256_storeu_pd(res_pointer + i + 12, _mm256_max_pd(_mm256_sub_pd(zero_vector, _mm256_loadu_pd(mat_pointer + i + 12)), _mm256_loadu_pd(mat_pointer + i + 12)));
        }
        
        //Tail Case
        #pragma omp parallel for
        for (int i = mat->rows * mat->cols / 16 * 16; i < mat->rows * mat->cols; i++) {  
            *(res_pointer + i) = fabs(*(mat_pointer + i));
        }
        return 0;

    }

    return -1;
}
