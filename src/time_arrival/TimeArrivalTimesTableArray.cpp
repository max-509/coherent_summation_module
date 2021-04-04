#include "TimeArrivalTimesTableArray.h"

TimeArrivalTimesTableArray::TimeArrivalTimesTableArray(
        py::array_t<double, py::array::c_style | py::array::forcecast> t_table) : t_table_(t_table) {
}

std::unique_ptr<float[]>
TimeArrivalTimesTableArray::get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) {
    std::unique_ptr<float[]> times_to_receivers;
    auto n_receivers = receivers_coords.size() / 3;
    auto n_points = t_table_.shape(0);
    try {
        times_to_receivers = std::make_unique<float[]>(n_points * n_receivers);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    auto t_table_unchecked = t_table_.unchecked<2>();

    if (is_receivers_inner) {
        #pragma omp parallel for collapse(2)
        for (auto i_p = 0; i_p < n_points; ++i_p) {
            for (auto i_r = current_receiver_idx_; i_r < current_receiver_idx_ + n_receivers; ++i_r) {
                times_to_receivers[i_p*n_receivers + (i_r - current_receiver_idx_)] = static_cast<float>(t_table_unchecked(i_p, i_r));
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (auto i_p = 0; i_p < n_points; ++i_p) {
            for (auto i_r = current_receiver_idx_; i_r < current_receiver_idx_ + n_receivers; ++i_r) {
                times_to_receivers[(i_r - current_receiver_idx_)*n_points + i_p] = static_cast<float>(t_table_unchecked(i_p, i_r));
            }
        }
    }

    current_receiver_idx_ += n_receivers;

    if (current_receiver_idx_ >= t_table_.shape(1)) {
        current_receiver_idx_ = 0;
    }

    return std::move(times_to_receivers);
}
