#include "CoherentSummation.h"
#include "time_arrival/TimeArrivalBase.h"
#include "utils.h"

#include <string>
#include <cstddef>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(CoherentSummationModule, coh_sum_module) {
    coh_sum_module.doc() = "Module for computing Kirchhoff Migration and Microseismic monitoring procedure"
                           "by emission tomography. Arrival times to depth points can be transferred by two ways:\n"
                           "1) By .csv-file with times;\n"
                           "2) By Neuron network: pb-file with frozen NN and path with NN model";

    py::enum_<NN_Type>(coh_sum_module, "NN_Type")
            .value("FROZEN", NN_Type::FROZEN)
            .value("MODEL", NN_Type::MODEL)
            .export_values();

    py::class_<CoherentSummation>(coh_sum_module, "CoherentSummation")
            .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, double, double, std::size_t, double, double, std::size_t, double, double, std::size_t>(),
                    "Constructor with time arrivals table array and environment {([z0, z1], z_dim,) ([x0, x1], x_dim), [optional for 3D ([y0, y1], y_dim)]}"
                    "t_table"_a,
                    "z0"_a, "z1"_a, "z_dim"_a,
                    "x0"_a, "x1"_a, "x_dim"_a,
                    "y0"_a=0.0, "y1"_a=0.0, "y_dim"_a=1)
            .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<double, py::array::c_style | py::array::forcecast>>(),
                 "Constructor with time arrivals table array and environment as numpy 2D or 3D array with coords (x, z) or (x, y, z)")
            .def(py::init<const std::string &, double, double, std::size_t, double, double, std::size_t, double, double, std::size_t>(),
                    "Constructor with time arrivals table filename and environment {([z0, z1], z_dim,) ([x0, x1], x_dim), [optional for 3D ([y0, y1], y_dim)]}"
                    "times_table_filename"_a,
                    "z0"_a, "z1"_a, "z_dim"_a,
                    "x0"_a, "x1"_a, "x_dim"_a,
                    "y0"_a=0.0, "y1"_a=0.0, "y_dim"_a=1)
            .def(py::init<const std::string &, py::array_t<double, py::array::c_style | py::array::forcecast>>(),
                 "Constructor with time arrivals table filename and environment as numpy 2D or 3D array with coords (x, z) or (x, y, z)")
            .def(py::init<const std::string &, NN_Type,
                         std::vector<std::pair<std::string, int>> &,
                         std::vector<std::pair<std::string, int>> &,
                         double, double, std::size_t,
                         double, double, std::size_t,
                         double, double, std::size_t>(),
                 "Constructor with NN: model in path or frozen filename as .pb format, input operations, output operation, grid of environment {([z0, z1], z_dim,) ([x0, x1], x_dim), [optional for 3D ([y0, y1], y_dim)]}",
                 "path_NN"_a, "nn_type"_a,
                 "input_ops"_a, "output_ops"_a,
                 "z0"_a, "z1"_a, "z_dim"_a,
                 "x0"_a, "x1"_a, "x_dim"_a,
                 "y0"_a=0.0, "y1"_a=0.0, "y_dim"_a=1)
            .def(py::init<const std::string &, NN_Type,
                         std::vector<std::pair<std::string, int>> &,
                         std::vector<std::pair<std::string, int>> &,
                         py::array_t<double, py::array::c_style | py::array::forcecast>>(),
                "Constructor with NN: model in path or frozen filename as .pb format, input operations, output operation and environment as numpy 2D or 3D array with coords (x, z) or (x, y, z)")
            .def("microseismic_monitoring", &CoherentSummation::emission_tomography_method, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
                    "gather"_a, "receivers_coords"_a, "dt"_a, "tensor_matrix"_a=cohSumUtils::create_diagonal_tensor_matrix());
}