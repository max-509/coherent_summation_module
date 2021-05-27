#ifndef PY_COMMON_H
#define PY_COMMON_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

template <typename T>
using py_array = py::array_t<T, py::array::c_style>;
using py_array_d = py_array<double>;
using py_array_f = py_array<float>;

using namespace py::literals;

#endif //PY_COMMON_H
