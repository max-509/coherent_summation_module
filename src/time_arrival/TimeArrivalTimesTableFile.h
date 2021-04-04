#ifndef _TIME_ARRIVAL_TIMES_TABLE_FILE_H
#define _TIME_ARRIVAL_TIMES_TABLE_FILE_H

#include "TimeArrivalBase.h"

#include <memory>
#include <string>

class TimeArrivalTimesTableFile : public TimeArrivalBase {
public:
    explicit TimeArrivalTimesTableFile(const std::string &table_filename, std::size_t n_points);

    std::unique_ptr<float[]>
    get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) override;

    ~TimeArrivalTimesTableFile() override;

private:
    const std::string table_filename_;
    std::size_t n_points_;
    std::size_t current_receiver_idx_ = 0;
};


#endif //_TIME_ARRIVAL_TIMES_TABLE_FILE_H
