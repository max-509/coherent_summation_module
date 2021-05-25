#ifndef _COHERENT_SUMMATION_UTILS_H
#define _COHERENT_SUMMATION_UTILS_H

#include "py_common.h"

namespace cohSumUtils {
    py::array_t<double, py::array::c_style | py::array::forcecast> create_diagonal_tensor_matrix();

    bool is_tensor_matrix_diag(py::array_t<double, py::array::c_style | py::array::forcecast> tensor_matrix);
}

#endif //_COHERENT_SUMMATION_UTILS_H