#include "matrix.h"
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

#define NUM_THREADS omp_get_thread_num()
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
void rand_matrix(matrix* result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data
 * array and initialize all entries to be zeros. Remember to set all fieds of
 * the matrix struct. `parent` should be set to NULL to indicate that this
 * matrix is not a slice. You should return -1 if either `rows` or `cols` or
 * both have invalid values, or if any call to allocate memory in this function
 * fails. If you don't set python error messages here upon failure, then
 * remember to set it in numc.c. Return 0 upon success and non-zero upon
 * failure.
 */
int allocate_matrix(matrix** mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "Nonpositive dimensions!");
        return -1;
    }

    (*mat) = (matrix*)malloc(sizeof(struct matrix));

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

    double* arr = calloc(rows * cols, sizeof(double));
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

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < rows; i++) {
        (*mat)->data[i] = &(arr[i * cols]);
    }
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and
 * `cols` columns. This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to
 * set it in numc.c. Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix** mat, matrix* from, int row_offset, int col_offset, int rows,
                        int cols) {
    if (rows <= 0 || cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "Negative index");
        return -1;
    }

    (*mat) = (matrix*)malloc(sizeof(struct matrix));

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

    // If from's parent is NULL, it is the root
    if (from->parent == NULL) {
        from->ref_cnt++;
        (*mat)->parent = from;
    } else {
        // If from's parent is not null, it is a child of parent
        from->parent->ref_cnt++;
        (*mat)->parent = from->parent;
    }

    // Ref_cnt does not really matter
    (*mat)->ref_cnt = 1;

    (*mat)->data = (double**)malloc(sizeof(double*) * rows);
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
 * This function will be called automatically by Python when a numc matrix
 * loses all of its reference pointers. You need to make sure that you only
 * free `mat->data` if no other existing matrices are also referring this data
 * array. See the spec for more information.
 */
void deallocate_matrix(matrix* mat) {
    /* Three scenarios:
     * -mat is NULL
     * -mat is the root
     *  -if ref_cnt <= 1, free EVERYTHING
     * -mat is child of root
     *  -if ref_cnt of root is <=1, free EVERYTHING as well as mat
     *  -if ref_cnt is > 1, just free the current mat, decrement ref_cnt
     */
    if (mat == NULL) {
        return;
    } else if (mat->parent == NULL && mat->ref_cnt <= 1) {
        // If mat is the last referencing matrix, free everything
        free(mat->data[0]);
        free(mat->data);
        free(mat);
        return;
    } else if (mat->parent != NULL && mat->parent->ref_cnt <= 1) { // && mat->parent->ref_cnt > 1
        // If mat is child of the root, then free matrix struct, decrement
        // ref_cnt, and nothing else
        free(mat->parent->data[0]);
        free(mat->parent->data);
        free(mat->parent);
        free(mat->data[0]);
        free(mat->data);
        free(mat);
        return;
    } else if (mat->parent != NULL && mat->parent->ref_cnt > 1) {
        mat->parent->ref_cnt--;
        free(mat->data);
        free(mat);
        return;
    } else {
        return;
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix* mat, int row, int col) { return mat->data[row][col]; }

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix* mat, int row, int col, double val) { mat->data[row][col] = val; }

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix* mat, double val) {
    memset(mat->data[0], val, mat->rows * mat->cols * sizeof(double));
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix* result, matrix* mat1, matrix* mat2) {
    if ((result->rows == mat1->rows && mat1->rows == mat2->rows) &&
        (result->cols == mat1->cols && mat1->cols == mat2->cols)) {
        if (mat1->rows < 16 || mat1->cols < 16 || mat1->parent != NULL || mat2->parent != NULL) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat1->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
                }
            }
            return 0;
        }

        double* res_pointer = result->data[0];
        double* mat1_pointer = mat1->data[0];
        double* mat2_pointer = mat2->data[0];

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < mat1->rows * mat1->cols / 16 * 16; i += 16) {
            _mm256_storeu_pd(res_pointer + i, _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i),
                                                            _mm256_loadu_pd(mat2_pointer + i)));
            _mm256_storeu_pd(res_pointer + i + 4,
                             _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 4),
                                           _mm256_loadu_pd(mat2_pointer + i + 4)));
            _mm256_storeu_pd(res_pointer + i + 8,
                             _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 8),
                                           _mm256_loadu_pd(mat2_pointer + i + 8)));
            _mm256_storeu_pd(res_pointer + i + 12,
                             _mm256_add_pd(_mm256_loadu_pd(mat1_pointer + i + 12),
                                           _mm256_loadu_pd(mat2_pointer + i + 12)));
        }

// Tail Case
#pragma omp parallel for num_threads(NUM_THREADS)
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
int sub_matrix(matrix* result, matrix* mat1, matrix* mat2) {
    if ((result->rows == mat1->rows && mat1->rows == mat2->rows) &&
        (result->cols == mat1->cols && mat1->cols == mat2->cols)) {
        if (mat1->rows < 16 || mat1->cols < 16 || mat1->parent != NULL || mat2->parent != NULL) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat1->cols; j++) {
                    result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
                }
            }
            return 0;
        }

        double* res_pointer = result->data[0];
        double* mat1_pointer = mat1->data[0];
        double* mat2_pointer = mat2->data[0];

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < mat1->rows * mat1->cols / 16 * 16; i += 16) {
            _mm256_storeu_pd(res_pointer + i, _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i),
                                                            _mm256_loadu_pd(mat2_pointer + i)));
            _mm256_storeu_pd(res_pointer + i + 4,
                             _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i + 4),
                                           _mm256_loadu_pd(mat2_pointer + i + 4)));
            _mm256_storeu_pd(res_pointer + i + 8,
                             _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i + 8),
                                           _mm256_loadu_pd(mat2_pointer + i + 8)));
            _mm256_storeu_pd(res_pointer + i + 12,
                             _mm256_sub_pd(_mm256_loadu_pd(mat1_pointer + i + 12),
                                           _mm256_loadu_pd(mat2_pointer + i + 12)));
        }

// Tail Case
#pragma omp parallel for num_threads(NUM_THREADS)
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
 * Remember that matrix multiplication is not the same as multiplying
 * individual elements.
 */
inline int mul_matrix(matrix* result, matrix* mat1, matrix* mat2) {
    if ((result->rows == mat1->rows) && (result->cols == mat2->cols) &&
        (mat1->cols == mat2->rows)) {
        double* data = (double*)calloc(mat1->rows * mat2->cols, sizeof(double));
        if (data == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed!");
            return 1;
        }

        double* result_pointer = result->data[0];
        int blocksize = 32;

        if (mat1->rows < blocksize || mat1->cols < blocksize || mat2->rows < blocksize ||
            mat2->cols < blocksize || mat1->parent != NULL || mat2->parent != NULL) {
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat2->cols; j++) {
                    for (int k = 0; k < mat2->rows; k++) {
                        *(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
                    }
                }
            }
            memcpy(result_pointer, data, result->rows * result->cols * sizeof(double));
            return 0;
        }

#pragma omp parallel for num_threads(NUM_THREADS) collapse(2) schedule(dynamic)
        for (int i_blocked = 0; i_blocked < mat1->rows; i_blocked += blocksize) {
            for (int j_blocked = 0; j_blocked < mat2->cols; j_blocked += blocksize) {
                for (int k_blocked = 0; k_blocked < mat2->rows; k_blocked += blocksize) {
                    for (int i = i_blocked; i < (i_blocked + blocksize) && i < mat1->rows; i++) {
                        for (int j = j_blocked;
                             j < j_blocked + blocksize && j < mat2->cols / 32 * 32; j += 32) {
                            register __m256d c0 = _mm256_loadu_pd(data + (i * mat2->cols) + j);
                            register __m256d c1 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 4);
                            register __m256d c2 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 8);
                            register __m256d c3 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 12);
                            register __m256d c4 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 16);
                            register __m256d c5 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 20);
                            register __m256d c6 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 24);
                            register __m256d c7 = _mm256_loadu_pd(data + (i * mat2->cols) + j + 28);

                            for (int k = k_blocked; k < k_blocked + blocksize && k < mat2->rows;
                                 k++) {
                                register __m256d mat1_reg = _mm256_broadcast_sd(mat1->data[i] + k);
                                c0 = _mm256_fmadd_pd(mat1_reg, _mm256_loadu_pd(mat2->data[k] + j),
                                                     c0);
                                c1 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 4), c1);
                                c2 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 8), c2);
                                c3 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 12), c3);
                                c4 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 16), c4);
                                c5 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 20), c5);
                                c6 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 24), c6);
                                c7 = _mm256_fmadd_pd(mat1_reg,
                                                     _mm256_loadu_pd(mat2->data[k] + j + 28), c7);
                            }

                            _mm256_storeu_pd(data + (i * mat2->cols) + j, c0);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 4, c1);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 8, c2);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 12, c3);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 16, c4);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 20, c5);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 24, c6);
                            _mm256_storeu_pd(data + (i * mat2->cols) + j + 28, c7);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < mat1->rows; i++) {
            for (int j = mat2->cols / 32 * 32; j < mat2->cols; j++) {
                for (int k = 0; k < mat2->rows; k++) {
                    *(data + (i * mat2->cols) + j) += mat1->data[i][k] * mat2->data[k][j];
                }
            }
        }

        memcpy(result_pointer, data, result->rows * result->cols * sizeof(double));
        free(data);
        return 0;
    }
    return -1;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise
 * multiplication.
 */
int pow_matrix(matrix* result, matrix* mat, int pow) {
    // Divide and conquer algorithm inspired from
    // https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/

    if (pow < 0) {
        PyErr_SetString(PyExc_ValueError, "Power is negative!");
        return -1;
    } else if (mat->rows != mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Matrix is not square!");
        return -1;
    } else {
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < result->rows; i++) {
            set(result, i, i, 1);
        }

        // Deep copy matrix data
        matrix* A = NULL;
        allocate_matrix(&A, mat->rows, mat->cols);
        if (mat->parent) {
            A->parent = mat->parent; // Copy the parent to switch to naive case in mul
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    A->data[i][j] = mat->data[i][j];
                }
            }
        } else {
            memcpy(A->data[0], mat->data[0], A->rows * A->cols * sizeof(double));
        }

        while (pow > 0) {
            if (pow & 1) {
                mul_matrix(result, A, result);
            }
            mul_matrix(A, A, A);
            pow /= 2;
        }

        A->parent = NULL;
        deallocate_matrix(A);
        return 0;
    }
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix* result, matrix* mat) {
    if ((result->rows == mat->rows) && (result->cols == mat->cols)) {
        if (mat->rows < 16 || mat->cols < 16 || mat->parent != NULL) {
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i][j] = (-1.0) * mat->data[i][j];
                }
            }
            return 0;
        }

        double* res_pointer = result->data[0];
        double* mat_pointer = mat->data[0];

        // -0.0 has the sign bit set and all other bits clear
        const __m256d sign_mask = _mm256_set1_pd(-0.0);

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < mat->rows * mat->cols / 32 * 32; i += 32) {
            __m256d vec0 = _mm256_loadu_pd(mat_pointer + i);
            __m256d vec1 = _mm256_loadu_pd(mat_pointer + i + 4);
            __m256d vec2 = _mm256_loadu_pd(mat_pointer + i + 8);
            __m256d vec3 = _mm256_loadu_pd(mat_pointer + i + 12);
            __m256d vec4 = _mm256_loadu_pd(mat_pointer + i + 16);
            __m256d vec5 = _mm256_loadu_pd(mat_pointer + i + 20);
            __m256d vec6 = _mm256_loadu_pd(mat_pointer + i + 24);
            __m256d vec7 = _mm256_loadu_pd(mat_pointer + i + 28);

            vec0 = _mm256_xor_pd(vec0, sign_mask);
            vec1 = _mm256_xor_pd(vec1, sign_mask);
            vec2 = _mm256_xor_pd(vec2, sign_mask);
            vec3 = _mm256_xor_pd(vec3, sign_mask);
            vec4 = _mm256_xor_pd(vec4, sign_mask);
            vec5 = _mm256_xor_pd(vec5, sign_mask);
            vec6 = _mm256_xor_pd(vec6, sign_mask);
            vec7 = _mm256_xor_pd(vec7, sign_mask);

            _mm256_storeu_pd(res_pointer + i, vec0);
            _mm256_storeu_pd(res_pointer + i + 4, vec1);
            _mm256_storeu_pd(res_pointer + i + 8, vec2);
            _mm256_storeu_pd(res_pointer + i + 12, vec3);
            _mm256_storeu_pd(res_pointer + i + 16, vec4);
            _mm256_storeu_pd(res_pointer + i + 20, vec5);
            _mm256_storeu_pd(res_pointer + i + 24, vec6);
            _mm256_storeu_pd(res_pointer + i + 28, vec7);
        }

// Tail Case
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = mat->rows * mat->cols / 32 * 32; i < mat->rows * mat->cols; i++) {
            *(res_pointer + i) = (-1.0) * *(mat_pointer + i);
        }
        return 0;
    }
    return 1;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix* result, matrix* mat) {
    if ((result->rows == mat->rows) && (result->cols == mat->cols)) {
        if (mat->rows < 16 || mat->cols < 16 || mat->parent != NULL) {
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i][j] = fabs(mat->data[i][j]);
                }
            }
            return 0;
        }

        // -0.0 has the sign bit set and all other bits cleared
        const __m256d sign_mask = _mm256_set1_pd(-0.0);
        double* res_pointer = result->data[0];
        double* mat_pointer = mat->data[0];

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < mat->rows * mat->cols / 16 * 16; i += 16) {
            __m256d vec0 = _mm256_loadu_pd(mat_pointer + i);
            __m256d vec1 = _mm256_loadu_pd(mat_pointer + i + 4);
            __m256d vec2 = _mm256_loadu_pd(mat_pointer + i + 8);
            __m256d vec3 = _mm256_loadu_pd(mat_pointer + i + 12);

            vec0 = _mm256_andnot_pd(sign_mask, vec0);
            vec1 = _mm256_andnot_pd(sign_mask, vec1);
            vec2 = _mm256_andnot_pd(sign_mask, vec2);
            vec3 = _mm256_andnot_pd(sign_mask, vec3);

            _mm256_storeu_pd(res_pointer + i, vec0);
            _mm256_storeu_pd(res_pointer + i + 4, vec1);
            _mm256_storeu_pd(res_pointer + i + 8, vec2);
            _mm256_storeu_pd(res_pointer + i + 12, vec3);
        }

// Tail Case
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = mat->rows * mat->cols / 16 * 16; i < mat->rows * mat->cols; i++) {
            *(res_pointer + i) = fabs(*(mat_pointer + i));
        }
        return 0;
    }
    return -1;
}