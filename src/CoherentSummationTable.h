#ifndef COHERENT_SUMMATION_TABLE_H
#define COHERENT_SUMMATION_TABLE_H

#include "py_common.h"

#include "CoherentSummation.h"

class TimeArrivalTimesTable;

struct TimeArrivalTimesTableDeleter {
    void operator()(TimeArrivalTimesTable *p);
};

class CoherentSummationTable final : public CoherentSummation {
public:

    explicit CoherentSummationTable(py_array_d t_table);

    CoherentSummationTable(const std::string &times_table_filename, py::ssize_t n_points);

    CoherentSummationTable(CoherentSummationTable &&) noexcept = default;
    CoherentSummationTable &operator=(CoherentSummationTable &&) noexcept = default;

    py_array_d
    emission_tomography_method(py_array_d gather,
                               py_array_d receivers_coords,
                               py_array_d sources_coords,
                               double dt,
                               std::ptrdiff_t i_r0,
                               std::ptrdiff_t i_rn,
                               py_array_d tensor_matrix,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    py_array_d
    emission_tomography_method(py_array_d gather,
                               std::ptrdiff_t i_r0,
                               std::ptrdiff_t i_rn,
                               double dt,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    static CoherentSummationTable
    createCoherentSummationTableArray(py_array_d table);

    static CoherentSummationTable
    createCoherentSummationTableFile(const std::string &times_table_filename, py::ssize_t n_points);

    ~CoherentSummationTable() noexcept override;

private:
    std::unique_ptr<TimeArrivalTimesTable, TimeArrivalTimesTableDeleter> time_arrival_nn_;
};


#endif //COHERENT_SUMMATION_TABLE_H
