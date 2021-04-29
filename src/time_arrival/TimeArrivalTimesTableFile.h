#ifndef _TIME_ARRIVAL_TIMES_TABLE_FILE_H
#define _TIME_ARRIVAL_TIMES_TABLE_FILE_H

#include "py_common.h"

#include "TimeArrivalBase.h"

#include <memory>
#include <string>

class TimeArrivalTimesTableFile final : public TimeArrivalBase {
public:
    TimeArrivalTimesTableFile() = default;

    explicit TimeArrivalTimesTableFile(std::string table_filename, std::size_t n_points);

    std::unique_ptr<float[]>
    get_times_to_points(std::ptrdiff_t i_r0, std::ptrdiff_t i_rn);

    std::size_t
    get_n_points() const;

    ~TimeArrivalTimesTableFile() noexcept override;

private:
    const std::string table_filename_{};
    std::size_t n_points_{};
};


#endif //_TIME_ARRIVAL_TIMES_TABLE_FILE_H
