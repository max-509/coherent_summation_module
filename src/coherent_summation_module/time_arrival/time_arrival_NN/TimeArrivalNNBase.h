#ifndef TIME_ARRIVAL_NN_BASE_H
#define TIME_ARRIVAL_NN_BASE_H

#include "tensorflow/c/c_api.h"

#include <vector>
#include <memory>

class TimeArrivalNNBase {
public:
    TimeArrivalNNBase();

    virtual float process(float x_p, float z_p, float x_r) = 0;

    virtual std::vector<float> process(std::vector<float> &x_p, std::vector<float> &z_p, std::vector<float> &x_r) = 0;

    virtual ~TimeArrivalNNBase() = default;

protected:
    std::shared_ptr<TF_Graph> graph_;
	std::shared_ptr<TF_Status> status_;
};


#endif //TIME_ARRIVAL_NN_BASE_H
