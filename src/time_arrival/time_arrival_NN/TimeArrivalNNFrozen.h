#ifndef _TIME_ARRIVAL_NN_FROZEN_H
#define _TIME_ARRIVAL_NN_FROZEN_H

#include "TimeArrivalNNBase.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstddef>
#include <tuple>

class TimeArrivalWrapperNN;

class TimeArrivalNNFrozen : public TimeArrivalNNBase {
public:
    TimeArrivalNNFrozen(const std::string &pb_filename,
                        std::vector<std::pair<std::string, int>> &input_ops,
                        std::vector<std::pair<std::string, int>> &output_ops) noexcept(false);

    float process(float, float, float) noexcept(false) final;

    std::vector<float> process(std::vector<float> &, std::vector<float> &, std::vector<float> &) noexcept(false) final;

    ~TimeArrivalNNFrozen() override = default;

private:

    std::shared_ptr<TF_ImportGraphDefOptions> graph_opts_;
    std::shared_ptr<TF_SessionOptions> session_opts_;
    std::shared_ptr<TF_Session> session_;
    std::vector<TF_Output> outputs_;
	std::vector<TF_Output> inputs_;

private:
    TF_Buffer *read_pb_file(const std::string &pb_filename);

};

#endif //_TIME_ARRIVAL_NN_FROZEN_H