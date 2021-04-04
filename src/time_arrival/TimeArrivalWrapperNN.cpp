#include "TimeArrivalWrapperNN.h"

#include "TimeArrivalNNBase.h"
#include "TimeArrivalNNFrozen.h"
#include "TimeArrivalNNModel.h"
#include "TimeArrivalNNException.h"

#include <iostream>

TimeArrivalWrapperNN::TimeArrivalWrapperNN(
        py::array_t<double, py::array::c_style | py::array::forcecast> environment, const std::string &pb_filename,
        std::vector<std::pair<std::string, int>> &input_ops, std::vector<std::pair<std::string, int>> &output_ops) :
        environment_(environment) {
    p_time_arrival_nn_ = std::make_unique<TimeArrivalNNFrozen>(pb_filename, input_ops, output_ops);
}

TimeArrivalWrapperNN::TimeArrivalWrapperNN(
        py::array_t<double, py::array::c_style | py::array::forcecast> environment, const std::string &model_path,
        const char *const *tags, int ntags, std::vector<std::pair<std::string, int>> &input_ops,
        std::vector<std::pair<std::string, int>> &output_ops) :
        environment_(environment) {
    p_time_arrival_nn_ = std::make_unique<TimeArrivalNNModel>(model_path, tags, ntags, input_ops, output_ops);
}

std::unique_ptr<float[]> TimeArrivalWrapperNN::get_times_to_receivers(std::vector<double> &receivers_coords, bool is_receivers_inner) {

    std::unique_ptr<float[]> times_to_receivers;
    try {
        std::size_t n_receivers = receivers_coords.size() / 3;
        auto n_points = environment_.size() / 3;
        times_to_receivers = std::make_unique<float[]>(n_points * n_receivers);
    } catch (const std::bad_alloc &bad_alloc_ex) {
        throw std::runtime_error(std::string("Error: cannot allocate memory with error: ") + bad_alloc_ex.what());
    }

    try {
        return get_times_to_receivers_3D_block(receivers_coords, is_receivers_inner, std::move(times_to_receivers));
    } catch (const TimeArrivalNNException &nn_ex3D) {
        std::cerr << "ANN error, try with less block size:" << nn_ex3D.what() << std::endl;
        try {
            return get_times_to_receivers_2D_block(receivers_coords, is_receivers_inner, std::move(times_to_receivers));
        } catch (const TimeArrivalNNException &nn_ex2D) {
            std::cerr << "ANN error, try with less block size:" << nn_ex2D.what() << std::endl;
            try {
                return get_times_to_receivers_1D_block(receivers_coords, is_receivers_inner,
                                                       std::move(times_to_receivers));
            } catch (const TimeArrivalNNException &nn_ex1D) {
                std::cerr << "ANN error, try with less block size: " << nn_ex1D.what() << std::endl;
                try {
                    return get_times_to_receivers_without_blocks(receivers_coords, is_receivers_inner,
                                                                 std::move(times_to_receivers));
                } catch (const TimeArrivalNNException &nn_ex_without_blocks) {
                    throw std::runtime_error(
                            std::string("ANN errors: ") + nn_ex3D.what() + nn_ex2D.what() + nn_ex1D.what() +
                            nn_ex_without_blocks.what());
                }
            }
        }
    }
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_receivers_3D_block(std::vector<double> &receivers_coords,
                                                          bool is_receivers_inner,
                                                          std::unique_ptr<float[]> &&times_to_receivers) {
    std::size_t n_receivers = receivers_coords.size() / 3;

    auto n_points = environment_.size() / 3;

    auto z_dim = environment_.shape()[0], y_dim = environment_.shape()[1], x_dim = environment_.shape()[2];

    auto raw_unchecked_environment = environment_.unchecked<4>();

    if (y_dim == 1) {
        if (is_receivers_inner) {
            std::vector<float> s_x_v(x_dim*n_receivers);
            std::vector<float> s_z_v(x_dim*n_receivers);
            std::vector<float> r_x_v(x_dim*n_receivers);
            #pragma omp parallel for
            for (auto i_x = 0; i_x < x_dim; ++i_x) {
                for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                    r_x_v[i_x*n_receivers + i_r] = static_cast<float>(receivers_coords[3*i_r + 0]);
                }
            }

            for (auto i_z = 0; i_z < x_dim; ++i_z) {
                for (auto i_x = 0; i_x < x_dim; ++i_x) {
                    std::fill(s_z_v.begin() + i_x*n_receivers,
                              s_z_v.begin() + (i_x+1)*n_receivers,
                              static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 2)));
                    std::fill(s_x_v.begin() + i_x*n_receivers,
                              s_x_v.begin() + (i_x+1)*n_receivers,
                              static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 0)));
                }
                auto times_to_receiver_by_z = p_time_arrival_nn_->process(s_x_v, s_z_v, r_x_v);
                std::copy(times_to_receiver_by_z.cbegin(),
                          times_to_receiver_by_z.cend(),
                          times_to_receivers.get() + i_z*x_dim*n_receivers);
            }
        } else {
            std::vector<float> s_x_v(z_dim*x_dim);
            std::vector<float> s_z_v(z_dim*x_dim);
            std::vector<float> r_x_v(z_dim*x_dim);

            for (auto i_z = 0; i_z < z_dim; ++i_z) {
                for (auto i_x = 0; i_x < x_dim; ++i_x) {
                    s_x_v[i_z*x_dim + i_x] = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 0));
                    s_z_v[i_z*x_dim + i_x] = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 2));
                }
            }
            for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                std::fill(r_x_v.begin(), r_x_v.end(), static_cast<float>(receivers_coords[i_r*3 + 0]));
                auto times_to_receiver_by_r = p_time_arrival_nn_->process(s_x_v, s_z_v, r_x_v);
                std::copy(times_to_receiver_by_r.cbegin(), times_to_receiver_by_r.cend(), times_to_receivers.get() + i_r*n_points);
            }
        }
    } else {
        //TODO: NN for 3D environment
    }

    return std::move(times_to_receivers);
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_receivers_2D_block(std::vector<double> &receivers_coords,
                                                          bool is_receivers_inner,
                                                          std::unique_ptr<float[]> &&times_to_receivers) {
    std::size_t n_receivers = receivers_coords.size() / 3;

    auto n_points = environment_.size() / 3;

    auto z_dim = environment_.shape()[0], y_dim = environment_.shape()[1], x_dim = environment_.shape()[2];

    auto raw_unchecked_environment = environment_.unchecked<4>();

    if (y_dim == 1) {
        if (is_receivers_inner) {
            std::vector<float> s_x_v(n_receivers);
            std::vector<float> s_z_v(n_receivers);
            std::vector<float> r_x_v(n_receivers);
            std::copy(receivers_coords.cbegin(), receivers_coords.cend(), r_x_v.begin());
            #pragma omp parallel for
            for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                r_x_v[i_r] = static_cast<float>(receivers_coords[i_r * 3 + 0]);
            }

            for (auto i_z = 0; i_z < x_dim; ++i_z) {
                for (auto i_x = 0; i_x < x_dim; ++i_x) {
                    std::fill(s_x_v.begin(), s_x_v.end(),static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 0)));
                    std::fill(s_z_v.begin(), s_z_v.end(),static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 2)));
                    auto times_to_receiver_by_z = p_time_arrival_nn_->process(s_x_v, s_z_v, r_x_v);
                    std::copy(times_to_receiver_by_z.cbegin(),
                          times_to_receiver_by_z.cend(),
                          times_to_receivers.get() + (i_z*x_dim + i_x)*n_receivers);
                }
            }
        } else {
            std::vector<float> s_x_v(x_dim);
            std::vector<float> s_z_v(x_dim);
            std::vector<float> r_x_v(x_dim);

            for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                std::fill(r_x_v.begin(), r_x_v.end(), static_cast<float>(receivers_coords[3*i_r+0]));
                for (auto i_z = 0; i_z < z_dim; ++i_z) {
                    for (auto i_x = 0; i_x < x_dim; ++i_x) {
                        s_x_v[i_x] = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 0));
                        s_z_v[i_x] = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 2));
                    }

                    auto times_to_receiver_by_r = p_time_arrival_nn_->process(s_x_v, s_z_v, r_x_v);
                    std::copy(times_to_receiver_by_r.cbegin(), times_to_receiver_by_r.cend(), times_to_receivers.get() + (i_r*z_dim + i_z)*x_dim);
                }
            }
        }
    } else {
        //TODO: NN for 3D environment
    }

    return std::move(times_to_receivers);
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_receivers_1D_block(std::vector<double> &receivers_coords,
                                                          bool is_receivers_inner,
                                                          std::unique_ptr<float[]> &&times_to_receivers) {
    std::size_t n_receivers = receivers_coords.size() / 3;

    auto n_points = environment_.size() / 3;

    auto z_dim = environment_.shape()[0], y_dim = environment_.shape()[1], x_dim = environment_.shape()[2];

    auto raw_unchecked_environment = environment_.unchecked<4>();

    if (y_dim == 1) {
        get_times_to_receivers_without_blocks(receivers_coords, is_receivers_inner, std::move(times_to_receivers));
    } else {
        //TODO: NN for 3D environment
    }

    return std::move(times_to_receivers);
}

std::unique_ptr<float[]>
TimeArrivalWrapperNN::get_times_to_receivers_without_blocks(std::vector<double> &receivers_coords,
                                                                bool is_receivers_inner,
                                                                std::unique_ptr<float[]> &&times_to_receivers) {
    std::size_t n_receivers = receivers_coords.size() / 3;

    auto n_points = environment_.size() / 3;

    auto z_dim = environment_.shape()[0], y_dim = environment_.shape()[1], x_dim = environment_.shape()[2];

    auto raw_unchecked_environment = environment_.unchecked<4>();

    if (y_dim == 1) {
        if (is_receivers_inner) {
            for (auto i_z = 0; i_z < z_dim; ++i_z) {
                for (auto i_x = 0; i_x < x_dim; ++i_x) {
                    auto z = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 2));
                    auto x = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 0));
                    for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                        auto r_x = static_cast<float>(receivers_coords[3*i_r+0]);
                        times_to_receivers[(i_z * x_dim + i_x) * n_receivers + i_r] = p_time_arrival_nn_->process(x, z,
                                                                                                                  r_x);
                    }
                }
            }
        } else {
            for (auto i_r = 0; i_r < n_receivers; ++i_r) {
                auto r_x = static_cast<float>(receivers_coords[i_r*3+0]);
                for (auto i_z = 0; i_z < z_dim; ++i_z) {
                    for (auto i_x = 0; i_x < x_dim; ++i_x) {
                        auto z = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 2));
                        auto x = static_cast<float>(raw_unchecked_environment(i_z, 0, i_x, 0));
                        times_to_receivers[(i_r * z_dim + i_z) * x_dim + i_x] = p_time_arrival_nn_->process(x, z,
                                                                                                                  r_x);
                    }
                }
            }
        }
    } else {
        //TODO: NN for 3D environment
    }

    return std::move(times_to_receivers);
}


