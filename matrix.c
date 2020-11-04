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
 * failure, then remember to set it in matrix.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows <= 0 || cols <= 0) {
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

    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    (*mat)->data = (double**) malloc(sizeof(double*) * rows);

    if ((*mat)->data == NULL) {
        free((*mat));
	    PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed.");
        return -1;
    }

    for (int i = 0; i < rows; i++) {
        (*mat)->data[i] = (double *) calloc(cols, sizeof(double));
        //Ask what to do when malloc/calloc fails in the inner loop.
    }

    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in matrix.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (row_offset > rows || col_offset > cols) {
        PyErr_SetString(PyExc_TypeError, "Out of bound slicing.");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
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

    from->ref_cnt += 1;
    (*mat)->ref_cnt = 0;
    (*mat)->parent = from;

    for (int i = 0; i < rows - row_offset + 1; i++) {
        (*mat)->data[i] = (double *) calloc(cols - col_offset, sizeof(double));
    }

    for (int i = 0; i < rows - row_offset + 1; i++) {
        for (int j = 0; j < cols - col_offset; j++) {
            (*mat)->data[i][j] = from->data[i + row_offset][j + col_offset];    
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
    if (mat == NULL) {
        return;
    }
    
    if (mat->parent == NULL && mat->ref_cnt <= 1) {
        free(mat->data);
        free(mat);
    } else if(mat->ref_cnt > 0) {
        mat->ref_cnt--;
    } else { 
        if (mat->parent->ref_cnt <= 1) {
            free(mat->parent->data);
            free(mat->parent);
        }
        free(mat);        
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
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat->data[i][j] = val;
        }
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
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
            }
        }
        return 0;
    }
    return 1;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((result->rows == mat1->rows && mat1->rows == mat2->rows) && 
        (result->cols == mat1->cols && mat1->cols == mat2->cols)) {
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat1->cols; j++) {
                result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
            }
        }
        return 0;
    }
    return 1;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((result->rows == mat1->rows) && (result->cols == mat2->cols) && (mat1->cols == mat2->rows)) { 
        
        double *data = (double*) malloc(sizeof(double) * mat1->rows * mat2->cols);
        if (data == NULL) {
            return 1;
        }

        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                int dot_product = 0;
                for (int k = 0; k < mat2->rows; k++) {
                    dot_product += mat1->data[i][k] * mat2->data[k][j];
                }
                *(data + (i * mat2->cols) + j) = dot_product;
            }
        }

        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                result->data[i][j] = *(data + (i * mat2->cols) + j);
            }
        }

        free(data);

        return 0;
    }
    return 1;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    /* Inspired from https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/
     */

    if (pow < 0 || (mat->rows != mat->cols)) {
        return 1;
    }

    allocate_matrix(&result, mat->rows, mat->cols);

    for (int i = 0; i < result->rows; i++) {
        set(result, i, i, 1);
    }
   
    while (pow > 0) {
        if (pow % 2 == 1) {
            mul_matrix(result, result, mat);
        }  
        mul_matrix(mat, mat, mat);
        pow = pow / 2;
    }
    

    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if ((result->rows == mat->rows) && (result->cols == mat->cols)) {
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                result->data[i][j] = (-1.0) * mat->data[i][j];
            }
        }
        return 0;
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
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) { 
                result->data[i][j] = abs(mat->data[i][j]);
            }
        }
        return 0;
    }
    return 1;
}
