#ifndef PY_COMMON_H
#define PY_COMMON_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

using py_array_d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using py_array_f = py::array_t<float, py::array::c_style | py::array::forcecast>;

using namespace py::literals;

#endif //PY_COMMON_H
