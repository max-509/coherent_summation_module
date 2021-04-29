#include "TimeArrivalTimesTableArray.h"

TimeArrivalTimesTableArray::TimeArrivalTimesTableArray(
        py::array_t<double, py::array::c_style | py::array::forcecast> t_table) : t_table_(t_table) {
}

std::unique_ptr<float[]>
TimeArrivalTimesTableArray::get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    std::unique_ptr<float[]> times_to_points;
    auto n_receivers = t_table_.shape(1);
    auto n_points = t_table_.shape(0);

    if (i_rn > n_receivers || i_r0 < 0) {
        throw std::runtime_error("Error: Bad receivers indexes");
    }

    auto times_to_points_size = (i_rn - i_r0);

    try {
        times_to_points = std::make_unique<float[]>(n_points * times_to_points_size);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    auto t_table_unchecked = t_table_.unchecked<2>();

    #pragma omp parallel for collapse(2)
    for (auto i_p = 0; i_p < n_points; ++i_p) {
        for (auto i_r = i_r0; i_r < i_rn; ++i_r) {
            times_to_points[i_p * times_to_points_size + (i_r - i_r0)] = static_cast<float>(t_table_unchecked(i_p, i_r));
        }
    }

    return std::move(times_to_points);
}

py::ssize_t TimeArrivalTimesTableArray::get_n_points() const {
    return t_table_.shape(0);
}
