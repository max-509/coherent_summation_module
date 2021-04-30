#ifndef _TIME_ARRIVAL_TIMES_TABLE_ARRAY_H
#define _TIME_ARRIVAL_TIMES_TABLE_ARRAY_H

#include "py_common.h"

#include "TimeArrivalBase.h"

#include <memory>
#include <vector>

class TimeArrivalTimesTableArray final : public TimeArrivalBase {
public:
    TimeArrivalTimesTableArray() = default;

    TimeArrivalTimesTableArray(py_array_d t_table);

    std::unique_ptr<float[]>
    get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn);

    py::ssize_t
    get_n_points() const;

private:
    py_array_d t_table_;
    py::ssize_t current_receiver_idx_ = 0;
};


#endif //_TIME_ARRIVAL_TIMES_TABLE_ARRAY_H
