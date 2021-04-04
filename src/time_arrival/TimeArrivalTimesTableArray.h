#ifndef _TIME_ARRIBAL_TIMES_TABLE_ARRAY_H
#define _TIME_ARRIBAL_TIMES_TABLE_ARRAY_H

#include "TimeArrivalBase.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <memory>
#include <vector>

namespace py = pybind11;


class TimeArrivalTimesTableArray : public TimeArrivalBase {
public:
    TimeArrivalTimesTableArray(py::array_t<double, py::array::c_style | py::array::forcecast> t_table);

    std::unique_ptr<float[]> get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) final;

private:
    py::array_t<double, py::array::c_style | py::array::forcecast> t_table_;
    py::ssize_t current_receiver_idx_ = 0;
};


#endif //_TIME_ARRIBAL_TIMES_TABLE_ARRAY_H
