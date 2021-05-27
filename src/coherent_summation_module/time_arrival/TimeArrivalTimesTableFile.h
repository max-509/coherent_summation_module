#ifndef _TIME_ARRIVAL_TIMES_TABLE_FILE_H
#define _TIME_ARRIVAL_TIMES_TABLE_FILE_H

#include "py_common.h"

#include <memory>
#include <string>
#include <cstddef>
#include <utility>
#include <vector>
#include <cmath>

#include "csv.hpp"
#include "TimeArrivalTimesTable.h"

using namespace csv;

template <typename T2>
class TimeArrivalTimesTableFile final : public TimeArrivalTimesTable {
public:
    TimeArrivalTimesTableFile() = default;

    explicit TimeArrivalTimesTableFile(std::string table_filename, py::ssize_t n_points);

    py_array<float>
    get_times_to_points_f(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) override;

    py_array<double>
    get_times_to_points_d(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) override;

    py_array<T2>
    get_times_to_points_impl(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn);

    py::ssize_t
    get_n_points() const override;

    ~TimeArrivalTimesTableFile() noexcept override = default;

private:
    const std::string table_filename_{};
    py::ssize_t n_points_{};
};


template<typename T2>
TimeArrivalTimesTableFile<T2>::TimeArrivalTimesTableFile(std::string table_filename, py::ssize_t n_points) :
        table_filename_(std::move(table_filename)),
        n_points_(n_points) {
}

template<typename T2>
py_array<float> TimeArrivalTimesTableFile<T2>::get_times_to_points_f(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    return get_times_to_points_impl(i_r0, i_rn);
}

template<typename T2>
py_array<double> TimeArrivalTimesTableFile<T2>::get_times_to_points_d(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    return get_times_to_points_impl(i_r0, i_rn);
}

template<typename T2>
py_array<T2>
TimeArrivalTimesTableFile<T2>::get_times_to_points_impl(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    CSVFormat format;
    format.delimiter(';').no_header().quote(false);
    CSVReader csv_reader(table_filename_.c_str(), format);

    py_array<T2> times_to_points;
    auto n_receivers = i_rn - i_r0;
    try {
        times_to_points = py_array<T2>({(ssize_t) n_points_, (ssize_t) n_receivers});
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    auto times_to_points_unchecked = times_to_points.template mutable_unchecked<2>();

    py::gil_scoped_release release;

    std::ptrdiff_t row_size;
    decltype(n_points_) i_p = 0;

    for (const auto &row : csv_reader) {
        row_size = row.size();
        #pragma omp parallel for
        for (auto i_r = i_r0; i_r < std::min(i_rn, row_size); ++i_r) {
            try {
                times_to_points_unchecked(i_p, (i_r - i_r0)) = row[i_r].get<float>();
            } catch (const std::runtime_error &r_ex) {
                std::cerr << "ERROR VALUE: " << row[i_r].get<std::string>() << std::endl;
                std::cerr << "POINT NUMBER: " << i_p << std::endl;
                std::cerr << "RECEIVER NUMBER: " << i_r << std::endl;
                std::cerr << r_ex.what() << std::endl;
            }
        }
        ++i_p;
    }

    py::gil_scoped_acquire acquire;

    return times_to_points;
}

template<typename T2>
py::ssize_t
TimeArrivalTimesTableFile<T2>::get_n_points() const {
    return n_points_;
}


#endif //_TIME_ARRIVAL_TIMES_TABLE_FILE_H
