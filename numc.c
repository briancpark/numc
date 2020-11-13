#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    /* TODO: YOUR CODE HERE */    
    if (!PyObject_TypeCheck(args, &Matrix61cType) || args == NULL ||
        !PyObject_TypeCheck(self, &Matrix61cType) || self == NULL) {
        //TypeError if not both a and b are of type numc.Matrix.
        PyErr_SetString(PyExc_TypeError, "Argument is not a matrix type!");
        return NULL;
    }

    if (PyObject_RichCompareBool(self->shape, ((Matrix61c*) args)->shape, Py_NE)) {
        //ValueError if a and b do not have the same dimensions.
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch!");
        return NULL;
    }

    matrix *sum = NULL;
    int allocate_error = allocate_matrix(&sum, self->mat->rows, self->mat->cols);

    if (allocate_error) {
        return NULL;
    }

    int add_error = add_matrix(sum, self->mat, ((Matrix61c*) args)->mat);

    if (add_error) {
        deallocate_matrix(sum);
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch!");
        return NULL;
    }

    Matrix61c *sum_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    sum_object->mat = sum;
    sum_object->shape = PyTuple_Pack(2, PyLong_FromLong(sum->rows), PyLong_FromLong(sum->cols));

    return (PyObject*) sum_object;
}

/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    if (!PyObject_TypeCheck(args, &Matrix61cType) || args == NULL ||
        !PyObject_TypeCheck(self, &Matrix61cType) || self == NULL) {
        //TypeError if not both a and b are of type numc.Matrix.
        PyErr_SetString(PyExc_TypeError, "Argument is not a matrix type!");
        return NULL;
    }

    if (PyObject_RichCompareBool(self->shape, ((Matrix61c*) args)->shape, Py_NE)) {
        //ValueError if a and b do not have the same dimensions.
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch!");
        return NULL;
    }

    matrix *difference = NULL;
    int allocate_error = allocate_matrix(&difference, self->mat->rows, self->mat->cols);

    if (allocate_error) {
        return NULL;
    }

    int sub_error = sub_matrix(difference, self->mat, ((Matrix61c*) args)->mat);

    if (sub_error) {
        deallocate_matrix(difference);
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch!");
        return NULL;
    }

    Matrix61c *difference_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    difference_object->mat = difference;
    difference_object->shape = PyTuple_Pack(2, PyLong_FromLong(difference->rows), PyLong_FromLong(difference->cols));

    return (PyObject*) difference_object;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    /* TODO: YOUR CODE HERE */
    if (!PyObject_TypeCheck(args, &Matrix61cType) || args == NULL ||
        !PyObject_TypeCheck(self, &Matrix61cType) || self == NULL) {
        //TypeError if not both a and b are of type numc.Matrix.
        PyErr_SetString(PyExc_TypeError, "Argument is not a matrix type!");
        return NULL;
    }

    if (self->mat->cols != ((Matrix61c*) args)->mat->rows) {
        //ValueError if a’s number of columns is not equal to b’s number of rows.
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch for matrix multiplication!");
        return NULL;
    }

    matrix *product = NULL;
    int allocate_error = allocate_matrix(&product, self->mat->rows, ((Matrix61c*) args)->mat->cols);

    if (allocate_error) {
        return NULL;
    }

    int mul_error = mul_matrix(product, self->mat, ((Matrix61c*) args)->mat);

    if (mul_error) {
        deallocate_matrix(product);
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch for matrix multiplication!");
        return NULL;
    }

    Matrix61c *product_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    product_object->mat = product;
    product_object->shape = PyTuple_Pack(2, PyLong_FromLong(product->rows), PyLong_FromLong(product->cols));

    return (PyObject*) product_object;
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    /* TODO: YOUR CODE HERE */
    if (!PyObject_TypeCheck(self, &Matrix61cType) || self == NULL) {
        //Very rare error, shouldn't really ever happen
        PyErr_SetString(PyExc_RuntimeError, "Nothing to negate!");
        return NULL;
    }

    matrix *neg = NULL;
    int allocate_error = allocate_matrix(&neg, self->mat->rows, self->mat->cols);

    if (allocate_error) {
        return NULL;
    }

    int neg_error = neg_matrix(neg, self->mat);

    if (neg_error) {
        deallocate_matrix(neg);
        return NULL;
    }

    Matrix61c *neg_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    neg_object->mat = neg;
    neg_object->shape = PyTuple_Pack(2, PyLong_FromLong(neg->rows), PyLong_FromLong(neg->cols));

    return (PyObject*) neg_object;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    /* TODO: YOUR CODE HERE */
    if (!PyObject_TypeCheck(self, &Matrix61cType) || self == NULL) {
        //Very rare error, shouldn't really ever happen
        PyErr_SetString(PyExc_RuntimeError, "Nothing to negate!");
        return NULL;
    }

    matrix *abs = NULL;
    int allocate_error = allocate_matrix(&abs, self->mat->rows, self->mat->cols);

    if (allocate_error) {
        return NULL;
    }

    int abs_error = abs_matrix(abs, self->mat);

    if (abs_error) {
        deallocate_matrix(abs);
        return NULL;
    }

    Matrix61c *abs_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    abs_object->mat = abs;
    abs_object->shape = PyTuple_Pack(2, PyLong_FromLong(abs->rows), PyLong_FromLong(abs->cols));

    return (PyObject*) abs_object;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    /* TODO: YOUR CODE HERE */
    if (!PyObject_TypeCheck(self, &Matrix61cType) || self == NULL) {
        //TypeError if a is not of type numc.Matrix or pow is not an integer.
        PyErr_SetString(PyExc_TypeError, "Argument is not a matrix type!");
        return NULL;
    }

    if (!PyLong_Check(pow)) {
        //TypeError if a is not of type numc.Matrix or pow is not an integer.
        PyErr_SetString(PyExc_TypeError, "Power is not an integer!");
        return NULL;
    }

    //ValueError if a is not a square matrix or if pow is negative.
    //Value errors are handled in the core matrix.c function!

    matrix *power = NULL;
    
    int allocate_error = allocate_matrix(&power, self->mat->rows, self->mat->cols);

    if (allocate_error) {
        return NULL;
    }

    int n = (int) PyLong_AsLong(pow);
    int pow_error = pow_matrix(power, self->mat, n);

    if (pow_error) {
        deallocate_matrix(power);
        return NULL;
    }

    Matrix61c *pow_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    pow_object->mat = power;
    pow_object->shape = PyTuple_Pack(2, PyLong_FromLong(power->rows), PyLong_FromLong(power->cols));

    return (PyObject*) pow_object;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    /* TODO: YOUR CODE HERE */
   .nb_add = (binaryfunc) &Matrix61c_add,
   .nb_subtract = (binaryfunc) &Matrix61c_sub,
   .nb_multiply = (binaryfunc) &Matrix61c_multiply,
   .nb_negative = (unaryfunc) &Matrix61c_neg,
   .nb_absolute = (unaryfunc) &Matrix61c_abs,
   .nb_power = (ternaryfunc) &Matrix61c_pow
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    if (self == NULL || args == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Arguments are null!");
        return Py_None;
    }

    PyObject *rows = NULL;
    PyObject *cols = NULL;
    PyObject *val = NULL;
    
    if (PyArg_UnpackTuple(args, "args", 3, 3, &rows, &cols, &val)) {
        // TypeError if the number of arguments parsed from args is not 3, 
        // if i and j are not integers, or if val is not a float or int.
        if (!PyLong_Check(rows) || !PyLong_Check(cols)) {
            PyErr_SetString(PyExc_TypeError, "Indices are not integers!");
            return Py_None;
        } 
        
        if (!PyLong_Check(val) && !PyFloat_Check(val)) {
            PyErr_SetString(PyExc_TypeError, "Value is not an integer or floating point number!");
            return Py_None;
        }

        if ((PyLong_AsLong(rows) >= self->mat->rows) || 
            (PyLong_AsLong(rows) < 0) || 
            (PyLong_AsLong(cols) >= self->mat->cols) || 
            (PyLong_AsLong(cols) < 0)) {
            PyErr_SetString(PyExc_IndexError, "Indices out of range!");
            return Py_None;
        }
        if (PyLong_Check(val)) {
            set(self->mat, (int) PyLong_AsLong(rows), (int) PyLong_AsLong(cols), PyLong_AsLong(val));
            return Py_None;    
        }
        set(self->mat, (int) PyLong_AsLong(rows), (int) PyLong_AsLong(cols), PyFloat_AsDouble(val));
        return Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of arguements!");
        return Py_None;
    }
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    if (self == NULL || args == NULL) {
        PyErr_SetString(PyExc_TypeError, "Arguments are null!");
        return NULL;
    }

    PyObject *rows = NULL;
    PyObject *cols = NULL;
    
    if (PyArg_UnpackTuple(args, "args", 1, 2, &rows, &cols)) {
        //TypeError if the number of arguments parsed from args 
        //is not 2 or if either i or j is not 
        if (!PyLong_Check(rows) || !PyLong_Check(cols)) {
            PyErr_SetString(PyExc_TypeError, "Indices are not integers!");
            return NULL;
        } 

        if ((PyLong_AsLong(rows) >= self->mat->rows) || 
            (PyLong_AsLong(rows) < 0) || 
            (PyLong_AsLong(cols) >= self->mat->cols) || 
            (PyLong_AsLong(cols) < 0)) {
            PyErr_SetString(PyExc_IndexError, "Indices out of range!");
            return NULL;
        } 
    
        double val = get(self->mat, (int) PyLong_AsLong(rows), (int) PyLong_AsLong(cols));
        return PyFloat_FromDouble(val);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguements!");
        return NULL;
    }
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    /* TODO: YOUR CODE HERE */
    {"set", (PyCFunction) Matrix61c_set_value, METH_VARARGS, "set"},
    {"get", (PyCFunction) Matrix61c_get_value, METH_VARARGS, "get"},
    {NULL, NULL, 0, NULL}
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    /* TODO: YOUR CODE HERE */
    /* Key can be a integer, tuple, or list of tuples
     *
     * How to handle slice object parsing was inspired from https://stackoverflow.com/questions/23214380/how-to-pass-a-tuple-of-slice-objects-to-c-via-the-python-c-api
     */

    matrix *slice = NULL;

    if (PyLong_Check(key)) {
        //When key is an int
        int allocate_ref_error;

        if (PyLong_AsLong(key) < 0 || (PyLong_AsLong(key) > self->mat->rows && !self->mat->is_1d)) {
            PyErr_SetString(PyExc_IndexError, "Indices out of range!");
            return NULL;
        }

        if (self->mat->is_1d && self->mat->rows == 1) {
            return PyFloat_FromDouble(get(self->mat, 0, PyLong_AsLong(key)));
        } else if (self->mat->is_1d && self->mat->cols == 1) {
            return PyFloat_FromDouble(get(self->mat, PyLong_AsLong(key), 0));
        } else {
            allocate_ref_error = allocate_matrix_ref(&slice, self->mat, PyLong_AsLong(key), 0, 1, self->mat->cols);
        }
        
        if (allocate_ref_error) {
            deallocate_matrix(slice);
            return NULL;
        }
    } else if (PySlice_Check(key)) {
        Py_ssize_t start = 0;
        Py_ssize_t stop = 0;
        Py_ssize_t step = 0;
        Py_ssize_t slicelength = 0;
        
        if (self->mat->is_1d && self->mat->rows == 1) {
            PySlice_GetIndicesEx(key, self->mat->cols, &start, &stop, &step, &slicelength);
            int allocate_ref_error = allocate_matrix_ref(&slice, self->mat, 0, start, self->mat->rows, stop - start);
            
            if (allocate_ref_error) {
                deallocate_matrix(slice);
                return NULL;
            }
        } else if (self->mat->is_1d && self->mat->cols == 1) {
            //A single slice
            PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength);
            int allocate_ref_error = allocate_matrix_ref(&slice, self->mat, start, 0, stop - start, self->mat->cols);
            
            if (allocate_ref_error) {
                deallocate_matrix(slice);
                return NULL;
            }
        } 
        if ((step == 0 && !self->mat->is_1d) || step > 1) {
            PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
            return NULL;
        }

    } else if (PyTuple_Check(key)){
        //When key is now a tuple of slices
        if (self->mat->is_1d) {
            PyErr_SetString(PyExc_TypeError, "1D matrices only support single slice!");
            return NULL;
        }
        
        PyObject *row_slice = NULL;
        PyObject *col_slice = NULL;
        if (!PyArg_UnpackTuple(key, "key", 2, 2, &row_slice, &col_slice)) {
            //TODO: Print a python error here
            return NULL;
        }

        Py_ssize_t row_start = 0;
        Py_ssize_t row_stop = 0;
        Py_ssize_t row_step = 0;
        Py_ssize_t row_slicelength = 0;

        Py_ssize_t col_start = 0;
        Py_ssize_t col_stop = 0;
        Py_ssize_t col_step = 0;
        Py_ssize_t col_slicelength = 0;

        if (PySlice_Check(row_slice)) {
            //Row is a slice type
            PySlice_GetIndicesEx(row_slice, self->mat->rows, &row_start, &row_stop, &row_step, &row_slicelength);
        } else { //PyLong_Check(row_slice) doesn't work, Ask a TA if we can inject undefined symbols a[1:?] etc.
            //Row is an int type
            row_start = PyLong_AsLong(row_slice);
            row_stop = row_start + 1;
        } 

        if (PySlice_Check(col_slice)) {
            //Row is a slice type
            PySlice_GetIndicesEx(col_slice, self->mat->cols, &col_start, &col_stop, &col_step, &col_slicelength);
        } else { //PyLong_Check(col_slice) doesn't work, Ask a TA if we can inject undefined symbols a[1:?] etc.
            //Row is an int type
            col_start = PyLong_AsLong(col_slice);
            col_stop = col_start + 1;
        } 

        if (row_step < 0 || row_step > 1 || col_step < 0 || col_step > 1) {
            PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
            return NULL;
        }

        if ((PyLong_Check(row_slice) && PyLong_Check(col_slice)) || 
            (row_stop - row_start == 1 && col_stop - col_start == 1)) {
            return PyFloat_FromDouble(get(self->mat, row_start, col_start));
        }

        int allocate_ref_error = allocate_matrix_ref(&slice, self->mat, row_start, col_start, row_stop - row_start, col_stop - col_start);
        
        if (allocate_ref_error) {
            deallocate_matrix(slice);
            return NULL;
        }
    } else {
        //A null case.... handle an error here later.
        return NULL;
    }

    Matrix61c *slice_object = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    slice_object->mat = slice;

    if (slice->rows == 1) {
        slice->is_1d = 1;
        slice_object->shape = PyTuple_Pack(1, PyLong_FromLong(slice->cols));
    } else if (slice->cols == 1) {
        slice->is_1d = 1;
        slice_object->shape = PyTuple_Pack(1, PyLong_FromLong(slice->rows));
    } else {
        slice_object->shape = PyTuple_Pack(2, PyLong_FromLong(slice->rows), PyLong_FromLong(slice->cols));
    }

    return (PyObject*) slice_object;
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    /* TODO: YOUR CODE HERE */
    if (PyLong_Check(key) && PyList_Check(v)) {
        //When key is an int and value is list
        for (int j = 0; j < PyList_Size(v); j++) {
            set(self->mat, PyLong_AsLong(key), j, PyFloat_AsDouble(PyList_GetItem(v, j)));
        }
    } else if (PyLong_Check(key) && (PyFloat_Check(v) || PyLong_Check(v))) {
        //A single slice and if matrix is 1d
        if (self->mat->is_1d && self->mat->rows == 1) {
            if (PyLong_AsLong(key) < 0 || PyLong_AsLong(key) >= self->mat->cols) {
                PyErr_SetString(PyExc_IndexError, "Indices out of range!");
                return -1;
            }
            set(self->mat, 0, PyLong_AsLong(key), PyFloat_AsDouble(v));
            return 0;
        } else if (self->mat->is_1d && self->mat->cols == 1) {
            //TODO: check implementation here!
            if (PyLong_AsLong(key) < 0 || PyLong_AsLong(key) >= self->mat->rows) {
                PyErr_SetString(PyExc_IndexError, "Indices out of range!");
                return -1;
            }
            set(self->mat, 0, PyLong_AsLong(key),  PyFloat_AsDouble(v));
            return 0;
        }
    } else if (PyTuple_Check(key)){
        PyObject *row_slice = NULL;
        PyObject *col_slice = NULL;
        if (!PyArg_UnpackTuple(key, "key", 2, 2, &row_slice, &col_slice)) {
            //TODO: Print a python error here
            return -1;
        }

        Py_ssize_t row_start = 0;
        Py_ssize_t row_stop = 0;
        Py_ssize_t row_step = 0;
        Py_ssize_t row_slicelength = 0;

        Py_ssize_t col_start = 0;
        Py_ssize_t col_stop = 0;
        Py_ssize_t col_step = 0;
        Py_ssize_t col_slicelength = 0;

        if (PySlice_Check(row_slice)) {
            //Row is a slice type
            PySlice_GetIndicesEx(row_slice, self->mat->rows, &row_start, &row_stop, &row_step, &row_slicelength);
        } else { //PyLong_Check(row_slice) doesn't work, Ask a TA if we can inject undefined symbols a[1:?] etc.
            //Row is an int type
            row_start = PyLong_AsLong(row_slice);
            row_stop = row_start + 1;
        } 

        if (PySlice_Check(col_slice)) {
            //Row is a slice type
            PySlice_GetIndicesEx(col_slice, self->mat->cols, &col_start, &col_stop, &col_step, &col_slicelength);
        } else { //PyLong_Check(col_slice) doesn't work, Ask a TA if we can inject undefined symbols a[1:?] etc.
            //Row is an int type
            col_start = PyLong_AsLong(col_slice);
            col_stop = col_start + 1;
        } 

        //Handle all the various slicing here!
        if (PyFloat_Check(v) || PyLong_Check(v)) {
            //Slicing replacement with just one singular value
            if (row_stop - row_start < self->mat->cols && col_stop - col_start < self->mat->rows) {
                if (row_stop - row_start == 1 && col_stop - col_start == 1) {
                    set(self->mat, row_start, col_start, PyFloat_AsDouble(v));
                } else {
                    PyErr_SetString(PyExc_TypeError, "Value is not valid!");
                    return -1;
                }
            } else {
                PyErr_SetString(PyExc_IndexError, "Indices out of range!");
                return -1;
            }
        } else if (PyList_Check(PyList_GetItem(v, 0))) {
            //Slicing replacement with 2d array
            for (int i = row_start; i < row_stop; i++) {
                for (int j = col_start; j < col_stop; j++) {
                    set(self->mat, i, j, PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(v, i - row_start), j - col_start)));
                }
            }
        } else if (PyList_Check(v) && !PyList_Check(PyList_GetItem(v, 0))) {
            //Slicing replacement with 1d rows/ cols

            //Handle rows
            if (PyList_Size(v) == (col_stop - col_start) && (row_stop - row_start) == 1) {
                int j = 0;
                for (int i = col_start; i < col_stop - col_start; i++) {
                    //for (int j = 0; j < PyList_Size(v); j++) {
                        set(self->mat, row_start, i,  PyFloat_AsDouble(PyList_GetItem(v, j)));
                    //}
                    j++;
                }
            //Handle cols
            } else if (PyList_Size(v) == (row_stop - row_start) && (col_stop - col_start) == 1) {
                int j = 0;
                for (int i = row_start; i < row_stop - row_start; i++) {
                    //for (int j = 0; j < PyList_Size(v); j++) {
                        set(self->mat, i, col_start, PyFloat_AsDouble(PyList_GetItem(v, j)));
                    //}
                    j++;
                }
            }
        } else {
            //Error handling here
        }
    } else {
        //A null case.... handle an error here later.
        return -1;
    }
    return 0;
}

PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}