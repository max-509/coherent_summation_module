#ifndef TIME_ARRIVAL_BASE_WRAPPER_NN_H
#define TIME_ARRIVAL_BASE_WRAPPER_NN_H

#include "TimeArrivalBase.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <vector>
#include <tuple>
#include <memory>
#include <string>

namespace py = pybind11;

class TimeArrivalNNBase;

class TimeArrivalWrapperNN : public TimeArrivalBase {
public:
    TimeArrivalWrapperNN(py::array_t<double, py::array::c_style | py::array::forcecast> environment,
                             const std::string &pb_filename,
                             std::vector<std::pair<std::string, int>> &input_ops,
                             std::vector<std::pair<std::string, int>> &output_ops);

    TimeArrivalWrapperNN(py::array_t<double, py::array::c_style | py::array::forcecast> environment,
                             const std::string &model_path,
                             char const *const *tags,
                             int ntags,
                             std::vector<std::pair<std::string, int>> &input_ops,
                             std::vector<std::pair<std::string, int>> &output_ops);


    std::unique_ptr<float[]>
    get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) noexcept(false) final;

    ~TimeArrivalWrapperNN() override = default;

private:
    py::array_t<double, py::array::c_style | py::array::forcecast> environment_;
    std::unique_ptr<TimeArrivalNNBase> p_time_arrival_nn_;

private:

    std::unique_ptr<float[]>
    get_times_to_receivers_3D_block(std::vector<double> &receivers_coords,
                                    bool is_receivers_inner,
                                    std::unique_ptr<float[]> &&times_to_receivers);

    std::unique_ptr<float[]>
    get_times_to_receivers_2D_block(std::vector<double> &receivers_coords,
                                    bool is_receivers_inner,
                                    std::unique_ptr<float[]> &&times_to_receivers);

    std::unique_ptr<float[]>
    get_times_to_receivers_1D_block(std::vector<double> &receivers_coords,
                                    bool is_receivers_inner,
                                    std::unique_ptr<float[]> &&times_to_receivers);

    std::unique_ptr<float[]>
    get_times_to_receivers_without_blocks(std::vector<double> &receivers_coords,
                                          bool is_receivers_inner,
                                          std::unique_ptr<float[]> &&times_to_receivers);
};


#endif //TIME_ARRIVAL_BASE_WRAPPER_NN_H
