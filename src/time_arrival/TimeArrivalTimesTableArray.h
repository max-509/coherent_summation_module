#ifndef _TIME_ARRIVAL_TIMES_TABLE_ARRAY_H
#define _TIME_ARRIVAL_TIMES_TABLE_ARRAY_H

#include "py_common.h"

#include <memory>
#include <vector>

#include "TimeArrivalTimesTable.h"

class TimeArrivalTimesTableArray final : public TimeArrivalTimesTable {
public:
    TimeArrivalTimesTableArray() = default;

    explicit TimeArrivalTimesTableArray(py_array_d t_table);

    std::unique_ptr<float[]>
    get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) override;

    py::ssize_t
    get_n_points() const override;

    ~TimeArrivalTimesTableArray() noexcept override = default;

private:
    py_array_d t_table_;
    py::ssize_t current_receiver_idx_ = 0;
};


#endif //_TIME_ARRIVAL_TIMES_TABLE_ARRAY_H
