#ifndef COHERENT_SUMMATION_TABLE_FILE_H
#define COHERENT_SUMMATION_TABLE_FILE_H

#include "py_common.h"

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <cstddef>

#include "CoherentSummation.h"
#include "TimeArrivalTimesTableFile.h"


class CoherentSummationTableFile final : public CoherentSummation {
public:
    CoherentSummationTableFile(const std::string &times_table_filename, std::size_t n_points);

    ~CoherentSummationTableFile() noexcept override;

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
private:
    TimeArrivalTimesTableFile time_arrival_nn_;
};


#endif //COHERENT_SUMMATION_TABLE_FILE_H
