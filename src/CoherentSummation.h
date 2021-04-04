#ifndef COHERENT_SUMMATION_TESTS_COHERENTSUMMATION_H
#define COHERENT_SUMMATION_TESTS_COHERENTSUMMATION_H

#include "utils.h"

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <cstddef>
#include "pybind11/numpy.h"

namespace py = pybind11;

enum NN_Type {
    FROZEN,
    MODEL
};

class TimeArrivalBase;

class CoherentSummation {
public:

    CoherentSummation(py::array_t<double, py::array::c_style | py::array::forcecast> t_table,
                      double z0, double z1, std::size_t z_dim,
                      double x0, double x1, std::size_t x_dim,
                      double y0 = 0.0, double y1 = 0.0, std::size_t y_dim = 1);

    CoherentSummation(py::array_t<double, py::array::c_style | py::array::forcecast> t_table,
                      py::array_t<double, py::array::c_style | py::array::forcecast> environment);

    CoherentSummation(const std::string &times_table_filename,
                      double z0, double z1, std::size_t z_dim,
                      double x0, double x1, std::size_t x_dim,
                      double y0 = 0.0, double y1 = 0.0, std::size_t y_dim = 1);

    CoherentSummation(const std::string &times_table_filename,
                        py::array_t<double, py::array::c_style | py::array::forcecast> environment);

    CoherentSummation(const std::string &path_NN, NN_Type nn_type,
                      std::vector<std::pair<std::string, int>> &input_ops,
                      std::vector<std::pair<std::string, int>> &output_ops,
                      double z0, double z1, std::size_t z_dim,
                      double x0, double x1, std::size_t x_dim,
                      double y0 = 0.0, double y1 = 0.0, std::size_t y_dim = 1);

    CoherentSummation(const std::string &path_NN, NN_Type nn_type,
                      std::vector<std::pair<std::string, int>> &input_ops,
                      std::vector<std::pair<std::string, int>> &output_ops,
                      py::array_t<double, py::array::c_style | py::array::forcecast> environment);

    py::array_t<double, py::array::c_style | py::array::forcecast>
    emission_tomography_method(const py::array_t<double, py::array::c_style | py::array::forcecast>& gather,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& receivers_coords,
                               double dt,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& tensor_matrix = cohSumUtils::create_diagonal_tensor_matrix());

private:
    py::array_t<double, py::array::c_style | py::array::forcecast> environment_;
    std::unique_ptr<TimeArrivalBase> p_time_arrival_;

private:
    CoherentSummation(double z0, double z1, std::size_t z_dim,
                      double x0, double x1, std::size_t x_dim,
                      double y0 = 0.0, double y1 = 0.0, std::size_t y_dim = 1);

    explicit CoherentSummation(py::array_t<double, py::array::c_style | py::array::forcecast> environment);
};


#endif //COHERENT_SUMMATION_TESTS_COHERENTSUMMATION_H
