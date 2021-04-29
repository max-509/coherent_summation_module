#include "CoherentSummationTableArray.h"

CoherentSummationTableArray::CoherentSummationTableArray(
        py::array_t<double, py::array::c_style | py::array::forcecast> t_table) : time_arrival_nn_(t_table) {}

py_array_d
CoherentSummationTableArray::emission_tomography_method(py_array_d gather,
                                                        py_array_d receivers_coords,
                                                        py_array_d sources_coords,
                                                        double dt,
                                                        std::ptrdiff_t i_r0,
                                                        std::ptrdiff_t i_rn,
                                                        py_array_d tensor_matrix,
                                                        std::ptrdiff_t receivers_block_size,
                                                        std::ptrdiff_t samples_block_size) {
    std::cout << "Start times table calculating" << std::endl;
    auto times_table = time_arrival_nn_.get_times_to_points(i_r0, i_rn);
    std::cout << "End times table calculating" << std::endl;

    auto n_points = time_arrival_nn_.get_n_points();
    auto n_receivers = i_rn - i_r0;

    py_array_f sources_receivers_times;

    {
        py::gil_scoped_acquire acquire;

        sources_receivers_times = py_array_f({(ssize_t)n_points, (ssize_t)n_receivers},
                                           {n_receivers * sizeof(float), sizeof(float)},
                                           times_table.get());
    }

    return CoherentSummation::emission_tomography_method(gather,
                                                         receivers_coords,
                                                         sources_coords,
                                                         sources_receivers_times,
                                                         dt,
                                                         tensor_matrix,
                                                         receivers_block_size,
                                                         samples_block_size);
}

py_array_d
CoherentSummationTableArray::emission_tomography_method(py_array_d gather,
                                                        std::ptrdiff_t i_r0,
                                                        std::ptrdiff_t i_rn,
                                                        double dt,
                                                        std::ptrdiff_t receivers_block_size,
                                                        std::ptrdiff_t samples_block_size) {
    std::cout << "Start times table calculating" << std::endl;
    auto times_table = time_arrival_nn_.get_times_to_points(i_r0, i_rn);
    std::cout << "End times table calculating" << std::endl;

    auto n_points = time_arrival_nn_.get_n_points();
    auto n_receivers = i_rn - i_r0;

    py_array_f sources_receivers_times;

    {
        py::gil_scoped_acquire acquire;

        sources_receivers_times = py_array_f({(ssize_t)n_points, (ssize_t)n_receivers},
                                           {n_receivers * sizeof(float), sizeof(float)},
                                           times_table.get());
    }

    return CoherentSummation::emission_tomography_method(gather,
                                                         sources_receivers_times,
                                                         dt,
                                                         receivers_block_size,
                                                         samples_block_size);
}

CoherentSummationTableArray::~CoherentSummationTableArray() noexcept = default;
