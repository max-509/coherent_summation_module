#include "TimeArrivalTimesTableFile.h"

#include "csv.hpp"

#include <string>
#include <cstddef>
#include <vector>

using namespace csv;

TimeArrivalTimesTableFile::TimeArrivalTimesTableFile(const std::string &table_filename, std::size_t n_points) :
        table_filename_(table_filename),
        n_points_(n_points) {
}

std::unique_ptr<float[]>
TimeArrivalTimesTableFile::get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) {
    CSVFormat format;
    format.delimiter(';').no_header().quote(false);
    CSVReader csv_reader(table_filename_.c_str(), format);

    std::unique_ptr<float[]> times_to_receivers;
    auto n_receivers = receivers_coords.size() / 3;
    try {
        times_to_receivers = std::make_unique<float[]>(n_points_ * n_receivers);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    std::size_t row_size = 0;

    if (is_receivers_inner) {
        decltype(n_points_) i_p = 0;
        for (const auto &row : csv_reader) {
            row_size = row.size();
            #pragma omp parallel for
            for (auto i_r = current_receiver_idx_; i_r < current_receiver_idx_ + n_receivers && i_r < row.size() ; ++i_r) {
                try {
                    times_to_receivers[i_p*n_receivers + (i_r - current_receiver_idx_)] = row[i_r].get<float>();
                } catch (const std::runtime_error& r_ex) {
                    std::cerr << "ERROR VALUE: " << row[i_r].get<std::string>() << std::endl;
                    std::cerr << "POINT NUMBER: " << i_p << std::endl;
                    std::cerr << "RECEIVER NUMBER: " << i_r << std::endl;
                    std::cerr << r_ex.what() << std::endl;
                }
            }
            ++i_p;
        }
    } else {
        decltype(n_points_) i_p = 0;
        for (const auto &row : csv_reader) {
            #pragma omp parallel for
            for (auto i_r = current_receiver_idx_; i_r < current_receiver_idx_ + n_receivers && i_r < row.size(); ++i_r) {
                times_to_receivers[(i_r - current_receiver_idx_) * n_points_ + i_p] = row[i_r].get<float>();
            }
            ++i_p;
        }
    }

    current_receiver_idx_ += n_receivers;

    if (current_receiver_idx_ >= row_size) {
        current_receiver_idx_ = 0;
    }

    return std::move(times_to_receivers);
}

TimeArrivalTimesTableFile::~TimeArrivalTimesTableFile() { }
