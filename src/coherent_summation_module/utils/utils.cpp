#include "utils.h"

#include <cmath>
#include <numeric>

namespace cohSumUtils {
    py::array_t<double, py::array::c_style | py::array::forcecast> create_diagonal_tensor_matrix() {
        auto diag_tensor_matrix = py::array_t<double>(6);
        auto raw_mutable_unchecked_diag_tensor_matrix = diag_tensor_matrix.mutable_unchecked<1>();
        raw_mutable_unchecked_diag_tensor_matrix(0) = 1.0;
        raw_mutable_unchecked_diag_tensor_matrix(1) = 1.0;
        raw_mutable_unchecked_diag_tensor_matrix(2) = 1.0;
        raw_mutable_unchecked_diag_tensor_matrix(3) = 0.0;
        raw_mutable_unchecked_diag_tensor_matrix(4) = 0.0;
        raw_mutable_unchecked_diag_tensor_matrix(5) = 0.0;

        return diag_tensor_matrix;
    }

    bool is_tensor_matrix_diag(py::array_t<double, py::array::c_style | py::array::forcecast> tensor_matrix) {
        auto unchecked_tensor_matrix = tensor_matrix.unchecked<1>();

        return std::abs(unchecked_tensor_matrix(0) - 1.0) < std::numeric_limits<double>::epsilon() &&
                std::abs(unchecked_tensor_matrix(1) - 1.0) < std::numeric_limits<double>::epsilon() &&
                std::abs(unchecked_tensor_matrix(2) - 1.0) < std::numeric_limits<double>::epsilon() &&
                std::abs(unchecked_tensor_matrix(3) - 0.0) < std::numeric_limits<double>::epsilon() &&
                std::abs(unchecked_tensor_matrix(4) - 0.0) < std::numeric_limits<double>::epsilon() &&
                std::abs(unchecked_tensor_matrix(5) - 0.0) < std::numeric_limits<double>::epsilon();
    }
}