#ifndef TIME_ARRIVAL_BASE_WRAPPER_NN_H
#define TIME_ARRIVAL_BASE_WRAPPER_NN_H

#include "py_common.h"

#include <vector>
#include <tuple>
#include <memory>
#include <string>

class TimeArrivalNNBase;

template <class T>
class Array1D;

template <class T>
class Array2D;

struct TimeArrivalNNBaseDeleter {
    void operator()(TimeArrivalNNBase *p);
};

class TimeArrivalWrapperNN {
public:
    TimeArrivalWrapperNN() = default;

    TimeArrivalWrapperNN(py_array_d environment,
                         const std::string &pb_filename,
                         std::vector<std::pair<std::string, int>> &input_ops,
                         std::vector<std::pair<std::string, int>> &output_ops);

    TimeArrivalWrapperNN(py_array_d environment,
                         const std::string &model_path,
                         char const *const *tags,
                         int ntags,
                         std::vector<std::pair<std::string, int>> &input_ops,
                         std::vector<std::pair<std::string, int>> &output_ops);

    TimeArrivalWrapperNN &operator=(TimeArrivalWrapperNN &&) noexcept = default;

    TimeArrivalWrapperNN(TimeArrivalWrapperNN &&) noexcept = default;

    ~TimeArrivalWrapperNN() noexcept;

    std::unique_ptr<float[]>
    get_times_to_points(const Array2D<double> &receivers_coords) noexcept(false);

    std::unique_ptr<float[]>
    get_times_to_point(const Array1D<double> &coord) noexcept(false);

private:

    py_array_d environment_{};
    std::unique_ptr<TimeArrivalNNBase, TimeArrivalNNBaseDeleter> p_time_arrival_nn_{};

private:

    static constexpr std::size_t MAX_GPU_SIZE = 1024*1024*1024;

    std::unique_ptr<float[]>
    get_times_to_points(const Array2D<double> &receivers_coords,
                        std::ptrdiff_t block_size,
                        std::unique_ptr<float[]> &&times_to_receivers);

    std::unique_ptr<float[]>
    get_times_to_point(const Array1D<double> &coord,
                       std::ptrdiff_t block_size,
                       std::unique_ptr<float[]> &&times_to_point);
};


#endif //TIME_ARRIVAL_BASE_WRAPPER_NN_H
