#ifndef _TIME_ARRIVAL_TIMES_TABLE_FILE_H
#define _TIME_ARRIVAL_TIMES_TABLE_FILE_H

#include "py_common.h"

#include <memory>
#include <string>

#include "TimeArrivalTimesTable.h"

class TimeArrivalTimesTableFile final : public TimeArrivalTimesTable {
public:
    TimeArrivalTimesTableFile() = default;

    explicit TimeArrivalTimesTableFile(std::string table_filename, py::ssize_t n_points);

    std::unique_ptr<float[]>
    get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn) override;

    py::ssize_t
    get_n_points() const override;

    ~TimeArrivalTimesTableFile() noexcept override = default;

private:
    const std::string table_filename_{};
    py::ssize_t n_points_{};
};


#endif //_TIME_ARRIVAL_TIMES_TABLE_FILE_H
