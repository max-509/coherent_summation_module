#include "TimeArrivalTimesTableFile.h"

#include "csv.hpp"

#include <string>
#include <cstddef>
#include <utility>
#include <vector>
#include <cmath>

using namespace csv;

TimeArrivalTimesTableFile::TimeArrivalTimesTableFile(std::string table_filename, py::ssize_t n_points) :
        table_filename_(std::move(table_filename)),
        n_points_(n_points) {
}

std::unique_ptr<float[]>
TimeArrivalTimesTableFile::get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) {
    CSVFormat format;
    format.delimiter(';').no_header().quote(false);
    CSVReader csv_reader(table_filename_.c_str(), format);

    std::unique_ptr<float[]> times_to_points;
    auto n_receivers = i_rn - i_r0;
    try {
        times_to_points = std::unique_ptr<float[]>(new float[n_points_ * n_receivers]);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    std::ptrdiff_t row_size{};

    decltype(n_points_) i_p = 0;
    for (const auto &row : csv_reader) {
        row_size = row.size();
        #pragma omp parallel for
        for (auto i_r = i_r0; i_r < std::min(i_rn, row_size); ++i_r) {
            try {
                times_to_points[i_p * n_receivers + (i_r - i_r0)] = row[i_r].get<float>();
            } catch (const std::runtime_error &r_ex) {
                std::cerr << "ERROR VALUE: " << row[i_r].get<std::string>() << std::endl;
                std::cerr << "POINT NUMBER: " << i_p << std::endl;
                std::cerr << "RECEIVER NUMBER: " << i_r << std::endl;
                std::cerr << r_ex.what() << std::endl;
            }
        }
        ++i_p;
    }

    return std::move(times_to_points);
}

py::ssize_t
TimeArrivalTimesTableFile::get_n_points() const {
    return n_points_;
}