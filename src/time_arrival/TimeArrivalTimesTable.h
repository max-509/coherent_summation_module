#ifndef TIME_ARRIVAL_TIMES_TABLE_H
#define TIME_ARRIVAL_TIMES_TABLE_H

#include "py_common.h"

#include <memory>
#include <cstddef>

class TimeArrivalTimesTable {
public:
    virtual ~TimeArrivalTimesTable() = default;

    virtual std::unique_ptr<float[]>
    get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) = 0;

    virtual py::ssize_t
    get_n_points() const = 0;
};


#endif //TIME_ARRIVAL_TIMES_TABLE_H
