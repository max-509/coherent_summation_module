#ifndef COHERENT_SUMMATION_TABLE_H
#define COHERENT_SUMMATION_TABLE_H

#include "py_common.h"

#include "CoherentSummation.h"
#include "TimeArrivalTimesTable.h"
#include "TimeArrivalTimesTableFile.h"
#include "TimeArrivalTimesTableArray.h"

class CoherentSummationTable final : public CoherentSummation {
public:

    template<typename T2>
    explicit CoherentSummationTable(py_array<T2> t_table);

    CoherentSummationTable(const std::string &times_table_filename,
                           py::ssize_t n_points,
                           py::dtype type);

    CoherentSummationTable(CoherentSummationTable &&) noexcept = default;

    CoherentSummationTable &operator=(CoherentSummationTable &&) noexcept = default;

    template<typename T1>
    py_array<T1>
    emission_tomography_method(py_array<T1> gather,
                               py_array<T1> receivers_coords,
                               py_array<T1> sources_coords,
                               double dt,
                               std::ptrdiff_t i_r0,
                               std::ptrdiff_t i_rn,
                               py_array<T1> tensor_matrix,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    template<typename T1>
    py_array<T1>
    emission_tomography_method(py_array<T1> gather,
                               std::ptrdiff_t i_r0,
                               std::ptrdiff_t i_rn,
                               double dt,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    ~CoherentSummationTable() noexcept override = default;

private:
    py::dtype type_;
    std::unique_ptr<TimeArrivalTimesTable> time_arrival_nn_;
};

template<typename T2>
CoherentSummationTable::CoherentSummationTable(py_array<T2> t_table) {
    type_ = py::dtype::of<T2>();
    time_arrival_nn_ = std::unique_ptr<TimeArrivalTimesTable>(new TimeArrivalTimesTableArray<T2>(t_table));
}

CoherentSummationTable::CoherentSummationTable(const std::string &times_table_filename,
                                               py::ssize_t n_points,
                                               py::dtype type) {
    type_ = type;
    if (type.is(py::dtype::of<float>())) {
        //...
        time_arrival_nn_ = std::unique_ptr<TimeArrivalTimesTable>(
                new TimeArrivalTimesTableFile<float>(times_table_filename, n_points));
    } else if (type.is(py::dtype::of<double>())) {
        //...
        time_arrival_nn_ = std::unique_ptr<TimeArrivalTimesTable>(
                new TimeArrivalTimesTableFile<double>(times_table_filename, n_points));
    } else {
        throw std::runtime_error("Error: dtype must be float32 or float64");
    }
}

template<typename T1>
py_array<T1>
CoherentSummationTable::emission_tomography_method(py_array<T1> gather, py_array<T1> receivers_coords,
                                                   py_array<T1> sources_coords, double dt, std::ptrdiff_t i_r0,
                                                   std::ptrdiff_t i_rn, py_array<T1> tensor_matrix,
                                                   std::ptrdiff_t receivers_block_size,
                                                   std::ptrdiff_t samples_block_size) {
    if (type_.is(py::dtype::of<float>())) {
        std::cout << "Start times table calculating" << std::endl;
        auto times_table = time_arrival_nn_->get_times_to_points_f(i_r0, i_rn);
        std::cout << "End times table calculating" << std::endl;
        return CoherentSummation::emission_tomography_method(gather,
                                                             receivers_coords,
                                                             sources_coords,
                                                             times_table,
                                                             dt,
                                                             tensor_matrix,
                                                             receivers_block_size,
                                                             samples_block_size);
    } else if (type_.is(py::dtype::of<double>())) {
        std::cout << "Start times table calculating" << std::endl;
        auto times_table = time_arrival_nn_->get_times_to_points_d(i_r0, i_rn);
        std::cout << "End times table calculating" << std::endl;
        return CoherentSummation::emission_tomography_method(gather,
                                                             receivers_coords,
                                                             sources_coords,
                                                             times_table,
                                                             dt,
                                                             tensor_matrix,
                                                             receivers_block_size,
                                                             samples_block_size);
    } else {
        throw std::runtime_error("Error: Type must be float32 or float64");
    }
}

template<typename T1>
py_array<T1>
CoherentSummationTable::emission_tomography_method(py_array<T1> gather,
                                                   std::ptrdiff_t i_r0, std::ptrdiff_t i_rn,
                                                   double dt,
                                                   std::ptrdiff_t receivers_block_size,
                                                   std::ptrdiff_t samples_block_size) {
    if (type_.is(py::dtype::of<float>())) {
        std::cout << "Start times table calculating" << std::endl;
        auto times_table = time_arrival_nn_->get_times_to_points_f(i_r0, i_rn);
        std::cout << "End times table calculating" << std::endl;
        return CoherentSummation::emission_tomography_method(gather,
                                                             times_table,
                                                             dt,
                                                             receivers_block_size,
                                                             samples_block_size);
    } else if (type_.is(py::dtype::of<double>())) {
        std::cout << "Start times table calculating" << std::endl;
        auto times_table = time_arrival_nn_->get_times_to_points_d(i_r0, i_rn);
        std::cout << "End times table calculating" << std::endl;
        return CoherentSummation::emission_tomography_method(gather,
                                                             times_table,
                                                             dt,
                                                             receivers_block_size,
                                                             samples_block_size);
    } else {
        throw std::runtime_error("Error: Type must be float32 or float64");
    }
}

template<typename T2>
CoherentSummationTable
createCoherentSummationTableArray(py_array<T2> table) {
    return CoherentSummationTable(table);
}

CoherentSummationTable
createCoherentSummationTableFile(const std::string &times_table_filename,
                                 py::ssize_t n_points,
                                 py::dtype type) {
    return CoherentSummationTable(times_table_filename, n_points, type);
}

#endif //COHERENT_SUMMATION_TABLE_H
