#ifndef COHERENT_SUMMATION_TABLE_ARRAY_H
#define COHERENT_SUMMATION_TABLE_ARRAY_H

#include "CoherentSummation.h"
#include "TimeArrivalTimesTableArray.h"

class CoherentSummationTableArray final : public CoherentSummation {
public:
    CoherentSummationTableArray(py::array_t<double, py::array::c_style | py::array::forcecast> t_table);

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

    ~CoherentSummationTableArray() noexcept override;
private:
    TimeArrivalTimesTableArray time_arrival_nn_;
};


#endif //COHERENT_SUMMATION_TABLE_ARRAY_H
