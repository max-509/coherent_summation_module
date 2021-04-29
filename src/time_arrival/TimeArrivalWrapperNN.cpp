#include "TimeArrivalWrapperNN.h"

#include "TimeArrivalNNBase.h"
#include "TimeArrivalNNFrozen.h"
#include "TimeArrivalNNModel.h"
#include "TimeArrivalNNException.h"

#include <iostream>

TimeArrivalWrapperNN::TimeArrivalWrapperNN(
        py_array_d environment, const std::string &pb_filename,
        std::vector<std::pair<std::string, int>> &input_ops, std::vector<std::pair<std::string, int>> &output_ops) :
        environment_(environment) {
    p_time_arrival_nn_ = std::make_unique<TimeArrivalNNFrozen>(pb_filename, input_ops, output_ops);
}

TimeArrivalWrapperNN::TimeArrivalWrapperNN(
        py_array_d environment, const std::string &model_path,
        const char *const *tags, int ntags, std::vector<std::pair<std::string, int>> &input_ops,
        std::vector<std::pair<std::string, int>> &output_ops) :
        environment_(environment) {
    p_time_arrival_nn_ = std::make_unique<TimeArrivalNNModel>(model_path, tags, ntags, input_ops, output_ops);
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_points(const Array2D<double> &receivers_coords) {

    std::unique_ptr<float[]> times_to_points;
    std::size_t n_receivers = receivers_coords.get_y_dim();
    auto n_points = environment_.shape(0);
    std::cerr << "N POINTS " << n_points << std::endl;
    try {
        times_to_points = std::unique_ptr<float[]>(new float[n_points * n_receivers]);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    auto block_size = n_points / 20;

    for (; block_size > 0; block_size /= 2) {
        std::cout << "Try block size:  " << block_size << std::endl;
        try {
            return get_times_to_points(receivers_coords, block_size, std::move(times_to_points));
        } catch (const TimeArrivalNNException &nn_ex) {
            std::cerr << "ANN error, try with less block size:" << nn_ex.what() << std::endl;
        }
    }

    throw std::runtime_error("Error: Cannot calculating times table by ANN");
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_point(const Array1D<double> &coord) noexcept(false) {
    std::unique_ptr<float[]> times_to_point;
    auto n_points = environment_.shape(0);
    try {
        times_to_point = std::unique_ptr<float[]>(new float[n_points]);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    auto block_size = n_points;

    for (; block_size > 0; block_size /= 2) {
        try {
            return get_times_to_point(coord, block_size, std::move(times_to_point));
        } catch (const TimeArrivalNNException &nn_ex) {
            std::cerr << "ANN error, try with less block size:" << nn_ex.what() << std::endl;
        }
    }

    throw std::runtime_error("Error: Cannot calculating times table by ANN");
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_point(const Array1D<double> &coord,
                                         std::ptrdiff_t block_size,
                                         std::unique_ptr<float[]> &&times_to_point) {

    auto n_points = environment_.shape(0);
    auto n_coords = coord.get_size();

    auto raw_unchecked_environment = environment_.unchecked<2>();

    if (n_coords == 1) {
        std::vector<float> x_coords(block_size, coord[0]);
        std::vector<float> x_points(block_size);
        std::vector<float> z_points(block_size);

        for (std::ptrdiff_t i_p = 0; i_p < (n_points - (n_points % block_size)); i_p += block_size) {
            for (auto i_bp = 0; i_bp < block_size; ++i_bp) {
                x_points[i_bp] = raw_unchecked_environment(i_p + i_bp, 0);
                z_points[i_bp] = raw_unchecked_environment(i_p + i_bp, 2);
            }

            auto times_to_receiver_by_z = p_time_arrival_nn_->process(x_points, z_points, x_coords);
            std::copy(times_to_receiver_by_z.cbegin(),
                      times_to_receiver_by_z.cend(),
                      times_to_point.get() + i_p);
        }

        x_coords.resize(n_points % block_size);
        x_points.resize(n_points % block_size);
        z_points.resize(n_points % block_size);

        for (auto i_bp = 0; i_bp < (n_points % block_size); ++i_bp) {
            x_points[i_bp] = raw_unchecked_environment((n_points - (n_points % block_size)) + i_bp, 0);
            z_points[i_bp] = raw_unchecked_environment((n_points - (n_points % block_size)) + i_bp, 2);
        }

        auto times_to_receiver_by_z = p_time_arrival_nn_->process(x_points, z_points, x_coords);
        std::copy(times_to_receiver_by_z.cbegin(),
                  times_to_receiver_by_z.cend(),
                  times_to_point.get() + (n_points - (n_points % block_size)));
    } else if (n_coords == 2) {
        //TODO: Для 3D
    } else if (n_coords == 3) {
        //TODO: Для 3D
    }

    return std::move(times_to_point);
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_points(const Array2D<double> &receivers_coords,
                                             std::ptrdiff_t block_size,
                                             std::unique_ptr<float[]> &&times_to_points) {
    auto n_points = environment_.shape(0);
    auto n_receivers = receivers_coords.get_y_dim();
    auto n_coords = receivers_coords.get_x_dim();

    auto raw_unchecked_environment = environment_.unchecked<2>();

    if (n_coords == 1) {


        std::vector<float> x_coords(block_size * n_receivers);
        std::vector<float> x_points(block_size * n_receivers);
        std::vector<float> z_points(block_size * n_receivers);

        for (std::ptrdiff_t i_bp = 0; i_bp < block_size; ++i_bp) {
            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                x_coords[i_bp * n_receivers + i_r] = receivers_coords(i_r, 0);
            }
        }

        for (std::ptrdiff_t i_p = 0; i_p < (n_points - (n_points % block_size)); i_p += block_size) {

            std::cerr << "I P: " << i_p << std::endl;

            for (auto i_bp = 0; i_bp < block_size; ++i_bp) {
                float x = raw_unchecked_environment(i_p + i_bp, 0);
                float z = raw_unchecked_environment(i_p + i_bp, 2);
                for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                    x_points[i_bp * n_receivers + i_r] = x;
                    z_points[i_bp * n_receivers + i_r] = z;
                }

            }

            auto times_to_receiver_by_z = p_time_arrival_nn_->process(x_points, z_points, x_coords);
            std::copy(times_to_receiver_by_z.cbegin(),
                      times_to_receiver_by_z.cend(),
                      times_to_points.get() + i_p * n_receivers);
        }

        if ((n_points % block_size) != 0) {
            std::vector<float> x_coords_remainder((n_points % block_size) * n_receivers);
            std::vector<float> x_points_remainder((n_points % block_size) * n_receivers);
            std::vector<float> z_points_remainder((n_points % block_size) * n_receivers);

            std::cerr << "X_COORDS_SIZE " << x_coords_remainder.size() <<std::endl;
            std::cerr << "x_points_SIZE " << x_points_remainder.size() <<std::endl;
            std::cerr << "z_points_SIZE " << z_points_remainder.size() <<std::endl;

            std::cerr << "LAST BLOCK SIZE : " << std::endl;
            std::cerr << (n_points % block_size) << std::endl;

            for (auto i_bp = 0; i_bp < (n_points % block_size); ++i_bp) {
                float x = raw_unchecked_environment((n_points - (n_points % block_size)) + i_bp, 0);
                float z = raw_unchecked_environment((n_points - (n_points % block_size)) + i_bp, 2);
                for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                    x_coords_remainder[i_bp * n_receivers + i_r] = receivers_coords(i_r, 0);
                    x_points_remainder[i_bp * n_receivers + i_r] = x;
                    z_points_remainder[i_bp * n_receivers + i_r] = z;

                    times_to_points[((n_points - (n_points % block_size)) + i_bp) * n_receivers + i_r] = p_time_arrival_nn_->process(x, z, receivers_coords(i_r, 0));
                }
            }

//            std::cerr << "LAST BLOCK ANN" << std::endl;
//
//            auto times_to_receiver_by_z = p_time_arrival_nn_->process(x_points_remainder, z_points_remainder, x_coords_remainder);
//            std::cerr << "AUTO processing last block" << std::endl;
//            std::copy(times_to_receiver_by_z.cbegin(),
//                      times_to_receiver_by_z.cend(),
//                      times_to_points.get() + (n_points - (n_points % block_size)) * n_receivers);
        }
    } else if (n_coords == 2) {
        //TODO: Для 3D
    } else if (n_coords == 3) {
        //TODO: Для 3D
    }

    return std::move(times_to_points);
}

TimeArrivalWrapperNN::~TimeArrivalWrapperNN() noexcept = default;


