#ifndef COHERENT_SUMMATION_H
#define COHERENT_SUMMATION_H

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <cstddef>


#include "utils.h"
#include "py_common.h"
#include "TimeArrivalBase.h"

class CoherentSummation {
public:

    CoherentSummation() = default;

    virtual ~CoherentSummation() noexcept;

//    py_array_d
//    emission_tomography_method(const py_array_d& gather,
//                               const py_array_d& receivers_coords,
//                               double dt,
//                               const py_array_d& tensor_matrix);

    py_array_d
    emission_tomography_method(py_array_d gather,
                               py_array_f sources_receivers_times,
                               double dt,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    py_array_d
    emission_tomography_method(py_array_d gather,
                               py_array_d receivers_coords,
                               py_array_d sources_coords,
                               py_array_f sources_receivers_times,
                               double dt,
                               py_array_d tensor_matrix,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    py_array_d
    kirchhoff_migration_method(py_array_d gather,
                              py_array_f times_to_source,
                              py_array_f times_to_receivers,
                              double dt);
};


#endif //COHERENT_SUMMATION_H
