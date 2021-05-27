#ifndef _COHERENT_SUMMATION_UTILS_H
#define _COHERENT_SUMMATION_UTILS_H

#include "py_common.h"

#include <cmath>
#include <numeric>

#include "array1D.h"
#include "array2D.h"

namespace cohSumUtils {
    template <typename T>
    py_array<T> create_diagonal_tensor_matrix() {
        auto diag_tensor_matrix = py_array<T>(6);
        auto raw_mutable_unchecked_diag_tensor_matrix = diag_tensor_matrix.template mutable_unchecked<1>();

        raw_mutable_unchecked_diag_tensor_matrix(0) = T(1.0);
        raw_mutable_unchecked_diag_tensor_matrix(1) = T(1.0);
        raw_mutable_unchecked_diag_tensor_matrix(2) = T(1.0);
        raw_mutable_unchecked_diag_tensor_matrix(3) = T(0.0);
        raw_mutable_unchecked_diag_tensor_matrix(4) = T(0.0);
        raw_mutable_unchecked_diag_tensor_matrix(5) = T(0.0);

        return diag_tensor_matrix;
    }

    template <typename T>
    bool is_tensor_matrix_diag(py_array<T> tensor_matrix) {
        auto unchecked_tensor_matrix = tensor_matrix.template unchecked<1>();

        return std::abs(unchecked_tensor_matrix(0) - T(1.0)) < std::numeric_limits<T>::epsilon() &&
                std::abs(unchecked_tensor_matrix(1) - T(1.0)) < std::numeric_limits<T>::epsilon() &&
                std::abs(unchecked_tensor_matrix(2) - T(1.0)) < std::numeric_limits<T>::epsilon() &&
                std::abs(unchecked_tensor_matrix(3) - T(0.0)) < std::numeric_limits<T>::epsilon() &&
                std::abs(unchecked_tensor_matrix(4) - T(0.0)) < std::numeric_limits<T>::epsilon() &&
                std::abs(unchecked_tensor_matrix(5) - T(0.0)) < std::numeric_limits<T>::epsilon();
    }

    template <typename T>
    Array2D<T> py_buffer_to_array2D(const py::buffer_info &array_info) {
        if (array_info.ndim != 2) {
            throw std::runtime_error("Error: Buffer ndims must be equal 2");
        }
        auto *array_p = static_cast<T *>(array_info.ptr);
        auto array_shape = array_info.shape;
        auto array_y_stride = (array_info.strides[0] / sizeof(T));
        auto array_x_stride = (array_info.strides[1] / sizeof(T));

        return Array2D<T>(array_p, array_shape[0], array_shape[1], array_y_stride, array_x_stride);
    }

    template <typename T>
    Array1D<T> py_buffer_to_array1D(const py::buffer_info &array_info) {
        if (array_info.ndim != 1) {
            throw std::runtime_error("Error: Buffer ndims must be equal 2");
        }
        auto *array_p = static_cast<T *>(array_info.ptr);
        auto array_shape = array_info.shape[0];
        auto array_stride = (array_info.strides[0] / sizeof(T));

        return Array1D<T>(array_p, array_shape, array_stride);
    }
}

#endif //_COHERENT_SUMMATION_UTILS_H