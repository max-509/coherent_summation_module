#include "py_common.h"

#include "CoherentSummation.h"
#include "CoherentSummationANN.h"
#include "CoherentSummationTable.h"

#include <string>
#include <cstddef>

#define DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR(Type1, Type2)                                            \
    def(                                                                                                \
        "emission_tomography",                                                                          \
        overload_cast_<py_array<Type1>, py_array<Type2>, double, std::ptrdiff_t, std::ptrdiff_t>()(     \
            &CoherentSummation::template emission_tomography_method<Type1, Type2>                       \
        ),                                                                                              \
        /*py::call_guard<py::gil_scoped_release>(), */                                                  \
        py::return_value_policy::move,                                                                  \
        "gather"_a,                                                                                     \
        "sources_receivers_times"_a,                                                                    \
        "dt"_a,                                                                                         \
        "receivers_block_size"_a = 20,                                                                  \
        "samples_block_size"_a = 1000                                                                   \
    )

#define DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR(Type1, Type2)                                               \
    def(                                                                                                \
        "emission_tomography",                                                                          \
        overload_cast_<py_array<Type1>, py_array<Type1>, py_array<Type1>, py_array<Type2>,              \
                        double, py_array<Type1>, std::ptrdiff_t, std::ptrdiff_t>()(                     \
            &CoherentSummation::template emission_tomography_method<Type1, Type2>                       \
        ),                                                                                              \
        /*py::call_guard<py::gil_scoped_release>(), */                                                  \
        py::return_value_policy::move,                                                                  \
        "gather"_a,                                                                                     \
        "receivers_coords"_a,                                                                           \
        "sources_coords"_a,                                                                             \
        "sources_receivers_times"_a,                                                                    \
        "dt"_a,                                                                                         \
        "tensor_matrix"_a,                                                                              \
        "receivers_block_size"_a = 20,                                                                  \
        "samples_block_size"_a = 1000                                                                   \
    )

#define DEF_KIRCHHOFF_MIGRATION_METHOD(Type1, Type2)                                                    \
    def(                                                                                                \
        "kirchhoff_migration",                                                                          \
        &CoherentSummation::template kirchhoff_migration_method<Type1, Type2>,                          \
        /*py::call_guard<py::gil_scoped_release>(), */                                                  \
        py::return_value_policy::move,                                                                  \
        "gather"_a,                                                                                     \
        "times_to_source"_a,                                                                            \
        "times_to_receivers"_a,                                                                         \
        "dt"_a,                                                                                         \
        "p_block_size"_a=0                                                                              \
    )

#define DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR_TABLE_FILE(Type1)                                           \
    def("emission_tomography",                                                                         \
        overload_cast_<                                                                                 \
            py_array<Type1>,                                                                            \
            py_array<Type1>,                                                                            \
            py_array<Type1>,                                                                            \
            double,                                                                                     \
            std::ptrdiff_t,                                                                             \
            std::ptrdiff_t,                                                                             \
            py_array<Type1>,                                                                            \
            std::ptrdiff_t,                                                                             \
            std::ptrdiff_t>()(                                                                          \
                &CoherentSummationTable::emission_tomography_method<Type1>                              \
        ),                                                                                              \
        /*py::call_guard<py::gil_scoped_release>(), */                                                  \
        py::return_value_policy::move,                                                                  \
        "gather"_a,                                                                                     \
        "receivers_coords"_a,                                                                           \
        "sources_coords"_a,                                                                             \
        "dt"_a, "i_r0"_a, "i_rn"_a,                                                                     \
        "tensor_matrix"_a,                                                                              \
        "receivers_block_size"_a = 20,                                                                  \
        "samples_block_size"_a = 1000)

#define DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR_TABLE_FILE(Type1)                                        \
    def("emission_tomography",                                                                          \
        overload_cast_<                                                                                 \
            py_array<Type1>,                                                                             \
            std::ptrdiff_t,                                                                             \
            std::ptrdiff_t,                                                                             \
            double,                                                                                     \
            std::ptrdiff_t,                                                                             \
            std::ptrdiff_t>()(                                                                          \
                &CoherentSummationTable::emission_tomography_method<Type1>                              \
        ),                                                                                              \
        /*py::call_guard<py::gil_scoped_release>(), */                                                  \
        py::return_value_policy::move,                                                                  \
        "gather"_a,                                                                                     \
        "i_r0"_a, "i_rn"_a, "dt"_a,                                                                     \
        "receivers_block_size"_a = 20,                                                                  \
        "samples_block_size"_a = 1000)



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
            .def(py::init<>())

            .DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR(double, double)

            .DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR(double, float)

            .DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR(float, double)

            .DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR(float, float)

            .DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR(double, double)

            .DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR(double, float)

            .DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR(float, double)

            .DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR(float, float)

            .DEF_KIRCHHOFF_MIGRATION_METHOD(double, double)

            .DEF_KIRCHHOFF_MIGRATION_METHOD(double, float)

            .DEF_KIRCHHOFF_MIGRATION_METHOD(float, double)

            .DEF_KIRCHHOFF_MIGRATION_METHOD(float, float);


    py::class_<CoherentSummationANN, CoherentSummation>(coh_sum_module, "CoherentSummationANN")
            .def(py::init(
                    [](const std::string &path_NN, NN_Type nn_type, std::vector<std::pair<std::string, int>> &input_ops,
                       std::vector<std::pair<std::string, int>> &output_ops,
                       py::tuple z_grid,
                       py::tuple x_grid,
                       py::tuple y_grid = py::make_tuple(0.0, 0.0, 1)) {
                        if (z_grid.size() != 3 || x_grid.size() != 3 || y_grid.size() != 3) {
                            throw std::invalid_argument("Error: Bad size of grid");
                        }

                        splitting_by_coord z_splitting = {z_grid[0].cast<double>(), z_grid[1].cast<double>(),
                                                          z_grid[2].cast<std::size_t>()};
                        splitting_by_coord x_splitting = {x_grid[0].cast<double>(), x_grid[1].cast<double>(),
                                                          x_grid[2].cast<std::size_t>()};
                        splitting_by_coord y_splitting = {y_grid[0].cast<double>(), y_grid[1].cast<double>(),
                                                          y_grid[2].cast<std::size_t>()};

                        return std::unique_ptr<CoherentSummationANN>(
                                new CoherentSummationANN(path_NN, nn_type, input_ops, output_ops, z_splitting,
                                                         x_splitting, y_splitting));
                    }),
                 "Constructor with NN: model in path or frozen filename as .pb format, input operations, output operation, grid of environment {([z0, z1], z_dim,) ([x0, x1], x_dim), [optional for 3D ([y0, y1], y_dim)]}",
                    /*py::call_guard<py::gil_scoped_release>(),*/
                 "path_NN"_a, "nn_type"_a,
                 "input_ops"_a, "output_ops"_a,
                 "z_tuple"_a, "x_tuple"_a, "y_tuple"_a = py::make_tuple(0.0, 0.0, 1))

            .def(py::init<const std::string &, NN_Type,
                    std::vector<std::pair<std::string, int>> &,
                    std::vector<std::pair<std::string, int>> &,
                    py_array_d>()/*,
                 py::call_guard<py::gil_scoped_release>()*/)

            .def("emission_tomography",
                 overload_cast_<py_array_d, py_array_d, double, py_array_d, std::ptrdiff_t, std::ptrdiff_t>()(
                         &CoherentSummationANN::emission_tomography_method),
                    /*py::call_guard<py::gil_scoped_release>(), */py::return_value_policy::move,
                 "gather"_a, "receivers_coords"_a, "dt"_a,
                 "tensor_matrix"_a, "receivers_block_size"_a = 20, "samples_block_size"_a = 1000)

            .def("emission_tomography",
                 overload_cast_<py_array_d, double, py_array_d, std::ptrdiff_t, std::ptrdiff_t>()(
                         &CoherentSummationANN::emission_tomography_method),
                    /*py::call_guard<py::gil_scoped_release>(), */py::return_value_policy::move,
                 "gather"_a, "dt"_a, "receivers_coords"_a,
                 "receivers_block_size"_a = 20, "samples_block_size"_a = 1000);

    py::class_<CoherentSummationTable, CoherentSummation>(coh_sum_module, "CoherentSummationTable")
            .def(py::init<py_array_d>()/*, py::call_guard<py::gil_scoped_release>()*/)

            .def(py::init<py_array_f>()/*, py::call_guard<py::gil_scoped_release>()*/)

            .def(py::init<const std::string &, py::ssize_t, py::dtype>()/*, py::call_guard<py::gil_scoped_release>()*/)

            .DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR_TABLE_FILE(double)

            .DEF_EMISSION_TOMOGRAPHY_WITH_TENSOR_TABLE_FILE(float)

            .DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR_TABLE_FILE(double)

            .DEF_EMISSION_TOMOGRAPHY_WITHOUT_TENSOR_TABLE_FILE(float);

    coh_sum_module.def("createCoherentSummationTableFile", createCoherentSummationTableFile);

    coh_sum_module.def("createCoherentSummationTableArray", createCoherentSummationTableArray<double>);

    coh_sum_module.def("createCoherentSummationTableArray", createCoherentSummationTableArray<float>);
}