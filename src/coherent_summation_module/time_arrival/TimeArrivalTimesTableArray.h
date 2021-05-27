#ifndef _TIME_ARRIVAL_TIMES_TABLE_ARRAY_H
#define _TIME_ARRIVAL_TIMES_TABLE_ARRAY_H

#include "py_common.h"

#include <memory>
#include <vector>

#include "TimeArrivalTimesTable.h"

template <typename T2>
class TimeArrivalTimesTableArray final : public TimeArrivalTimesTable {
public:
    TimeArrivalTimesTableArray() = default;

    explicit TimeArrivalTimesTableArray(py_array<T2> t_table);

    py_array<float>
    get_times_to_points_f(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) override;

    py_array<double>
    get_times_to_points_d(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) override;

    py_array<T2>
    get_times_to_points_impl(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn);

    py::ssize_t
    get_n_points() const override;

    ~TimeArrivalTimesTableArray() noexcept override = default;

private:
    py_array<T2> t_table_;
    py::ssize_t current_receiver_idx_ = 0;
};


template <typename T2>
TimeArrivalTimesTableArray<T2>::TimeArrivalTimesTableArray(py_array<T2> t_table) : t_table_(t_table) { }

template<typename T2>
py_array<float> TimeArrivalTimesTableArray<T2>::get_times_to_points_f(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    return get_times_to_points_impl(i_r0, i_rn);
}

template<typename T2>
py_array<double> TimeArrivalTimesTableArray<T2>::get_times_to_points_d(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    return get_times_to_points_impl(i_r0, i_rn);
}

template <typename T2>
py_array<T2>
TimeArrivalTimesTableArray<T2>::get_times_to_points_impl(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    auto t_table_info = t_table_.request();

    auto n_receivers = t_table_info.shape[1];
    auto n_points = t_table_info.shape[0];

    if (i_rn > n_receivers || i_r0 < 0) {
        throw std::runtime_error("Error: Bad receivers indexes");
    }

    auto times_to_points_size = (i_rn - i_r0);

    py_array<T2> times_to_points({(ssize_t) n_points, (ssize_t) times_to_points_size},
                                       t_table_info.strides,
                                       static_cast<T2*>(t_table_info.ptr) + i_r0,
                                       t_table_);

    return times_to_points;
}

template <typename T2>
py::ssize_t TimeArrivalTimesTableArray<T2>::get_n_points() const {
    return t_table_.shape(0);
}


#endif //_TIME_ARRIVAL_TIMES_TABLE_ARRAY_H
