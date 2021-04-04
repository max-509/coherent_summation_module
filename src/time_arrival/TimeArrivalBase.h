#ifndef _TIME_ARRIVAL_BASE_H
#define _TIME_ARRIVAL_BASE_H

#include <memory>
#include <vector>

class TimeArrivalBase {
public:

    virtual std::unique_ptr<float[]> get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) = 0;

    virtual ~TimeArrivalBase() = default;
};


#endif //_TIME_ARRIVAL_BASE_H
