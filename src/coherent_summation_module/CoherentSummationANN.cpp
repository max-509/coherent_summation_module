#include "CoherentSummationANN.h"

#include "array2D.h"

#include <type_traits>

CoherentSummationANN::CoherentSummationANN(const std::string &path_NN, NN_Type nn_type,
                                           std::vector<std::pair<std::string, int>> &input_ops,
                                           std::vector<std::pair<std::string, int>> &output_ops,
                                           splitting_by_coord x_splitting,
                                           splitting_by_coord z_splitting,
                                           splitting_by_coord y_splitting) {
    generate_environment_grid(x_splitting, z_splitting, y_splitting);


    load_ann(path_NN, nn_type, input_ops, output_ops);
}

CoherentSummationANN::CoherentSummationANN(const std::string &path_NN, NN_Type nn_type,
                                           std::vector<std::pair<std::string, int>> &input_ops,
                                           std::vector<std::pair<std::string, int>> &output_ops,
                                           py_array_d environment) {

    copy_environment(environment);


    load_ann(path_NN, nn_type, input_ops, output_ops);
}

void CoherentSummationANN::generate_environment_grid(splitting_by_coord x_splitting,
                                                     splitting_by_coord z_splitting,
                                                     splitting_by_coord y_splitting) {

    double x0 = x_splitting.coord_0, x1 = x_splitting.coord_1;
    auto x_dim = x_splitting.coord_dim;
    double z0 = z_splitting.coord_0, z1 = z_splitting.coord_1;
    auto z_dim = z_splitting.coord_dim;
    double y0 = y_splitting.coord_0, y1 = x_splitting.coord_1;
    auto y_dim = y_splitting.coord_dim;

    double dz = (z1 - z0) / static_cast<double>(z_dim - 1);
    double dy;
    if (y_dim == 1) {
        dy = 0.0;
    } else {
        dy = (y1 - y0) / static_cast<double>(y_dim - 1);
    }
    double dx = (x1 - x0) / static_cast<double>(x_dim - 1);

    environment_ = py_array_d(
            std::vector<py::ssize_t>{(py::ssize_t) z_dim, (py::ssize_t) y_dim, (py::ssize_t) x_dim, 3});

    auto raw_unchecked_mutable_environment = environment_.mutable_unchecked<4>();

#pragma omp parallel for collapse(3)
    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
        for (std::ptrdiff_t i_y = 0; i_y < y_dim; ++i_y) {
            for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {
                double z = z0 + static_cast<double>(i_z) * dz;
                double y = y0 + static_cast<double>(i_y) * dy;
                double x = x0 + static_cast<double>(i_x) * dx;
                raw_unchecked_mutable_environment(i_z, i_y, i_x, 0) = x;
                raw_unchecked_mutable_environment(i_z, i_y, i_x, 1) = y;
                raw_unchecked_mutable_environment(i_z, i_y, i_x, 2) = z;
            }
        }
    }

    environment_.resize(std::vector<py::ssize_t>{(py::ssize_t)(z_dim * y_dim * x_dim), 3});

}

void CoherentSummationANN::copy_environment(py_array_d environment) {

    if (environment.ndim() == 1) {
        throw std::runtime_error("Error: Bad environment shape");
    } else if (environment.ndim() == 2) {
        auto n_points = environment.shape(0);

        if (environment.shape(1) < 2) {
            throw std::runtime_error("Error: Bad environment shape");
        } else if (environment.shape(1) == 2) {
            environment_ = py::array_t<double>(std::vector<py::ssize_t>{n_points, 3});

            auto raw_environment_ = environment_.mutable_unchecked<2>();
            auto raw_environment = environment.unchecked<2>();

            py::gil_scoped_release release;

#pragma omp parallel for
            for (auto i_p = 0; i_p < n_points; ++i_p) {
                raw_environment_(i_p, 0) = raw_environment(i_p, 0);
                raw_environment_(i_p, 1) = 0.0;
                raw_environment_(i_p, 2) = raw_environment(i_p, 1);
            }

            py::gil_scoped_acquire acquire;

        } else if (environment.shape(1) == 3) {
            environment_ = py::array_t<double>(std::vector<py::ssize_t>{n_points, 3});

            auto raw_environment_ = environment_.mutable_unchecked<2>();
            auto raw_environment = environment.unchecked<2>();

            py::gil_scoped_release release;

#pragma omp parallel for
            for (auto i_p = 0; i_p < n_points; ++i_p) {
                raw_environment_(i_p, 0) = raw_environment(i_p, 0);
                raw_environment_(i_p, 1) = raw_environment(i_p, 1);
                raw_environment_(i_p, 2) = raw_environment(i_p, 2);
            }

            py::gil_scoped_acquire acquire;
        }
    } else if (environment.ndim() == 3) {
        environment_ = py_array_d(std::vector<py::ssize_t>{environment.shape(0), 1, environment.shape(1), 3});
        auto raw_environment_ = environment_.mutable_unchecked<4>();
        auto raw_environment = environment.unchecked<3>();

        py::gil_scoped_release release;

#pragma omp parallel for collapse(2)
        for (auto i_z = 0; i_z < environment.shape(0); ++i_z) {
            for (auto i_x = 0; i_x < environment.shape(1); ++i_x) {
                raw_environment_(i_z, 0, i_x, 0) = raw_environment(i_z, i_x, 0);
                raw_environment_(i_z, 0, i_x, 1) = 0.0;
                raw_environment_(i_z, 0, i_x, 2) = raw_environment(i_z, i_x, 1);
            }
        }

        py::gil_scoped_acquire acquire;

        environment_.resize(std::vector<py::ssize_t>{environment.shape(0) * environment.shape(1), 3});
    } else if (environment.ndim() == 4) {
        environment_ = py_array_d(
                std::vector<py::ssize_t>{environment.shape(0), environment.shape(1), environment.shape(2),
                                         environment.shape(3)});

        auto raw_environment_ = environment_.mutable_unchecked<4>();
        auto raw_environment = environment.unchecked<4>();

        py::gil_scoped_release release;

#pragma omp parallel for collapse(3)
        for (auto i_z = 0; i_z < environment.shape(0); ++i_z) {
            for (auto i_y = 0; i_y < environment.shape(1); ++i_y) {
                for (auto i_x = 0; i_x < environment.shape(2); ++i_x) {
                    raw_environment_(i_z, i_y, i_x, 0) = raw_environment(i_z, i_y, i_x, 0);
                    raw_environment_(i_z, i_y, i_x, 1) = raw_environment(i_z, i_y, i_x, 1);
                    raw_environment_(i_z, i_y, i_x, 2) = raw_environment(i_z, i_y, i_x, 2);
                }
            }
        }

        py::gil_scoped_acquire acquire;

        environment_.resize(
                std::vector<py::ssize_t>{environment.shape(0) * environment.shape(1) * environment.shape(2), 3});
    } else {
        throw std::runtime_error("Error: Bad environment shape");
    }

}

void CoherentSummationANN::load_ann(const std::string &path_NN, NN_Type nn_type,
                                    std::vector<std::pair<std::string, int>> &input_ops,
                                    std::vector<std::pair<std::string, int>> &output_ops) {
    switch (nn_type) {
        case NN_Type::FROZEN:
            time_arrival_nn_ = std::move(TimeArrivalWrapperNN(environment_, path_NN, input_ops, output_ops));
            break;
        case NN_Type::MODEL:
            const char *tags = "serve";
            time_arrival_nn_ = std::move(TimeArrivalWrapperNN(environment_, path_NN, &tags, 1, input_ops, output_ops));
            break;
    }
}

py_array_d CoherentSummationANN::emission_tomography_method(py_array_d gather,
                                                            py_array_d receivers_coords,
                                                            double dt,
                                                            py_array_d tensor_matrix,
                                                            std::ptrdiff_t receivers_block_size,
                                                            std::ptrdiff_t samples_block_size) {
    auto receivers_coords_info = receivers_coords.request();
    auto *receivers_coords_data = static_cast<double *>(receivers_coords_info.ptr);

    auto n_receivers = receivers_coords_info.shape[0];
    Array2D<double> receivers_coords2D;

    if (receivers_coords_info.ndim == 1) {
        receivers_coords2D = Array2D<double>(receivers_coords_data, n_receivers, 1,
                                             receivers_coords.strides()[0] / sizeof(double), 1);
    } else {
        receivers_coords2D = cohSumUtils::py_buffer_to_array2D<double>(receivers_coords.request());
    }

    std::cout << "Start times table calculating" << std::endl;
    auto times_table = time_arrival_nn_.get_times_to_points(receivers_coords2D);
    std::cout << "End times table calculating" << std::endl;

    auto n_points = environment_.shape(0);

    float *times_table_p = times_table.release();

    py::capsule free_when_done(times_table_p, [](void *f) {
        auto *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    py_array_f sources_receivers_times({n_points, n_receivers},
                                       {n_receivers * sizeof(float), sizeof(float)},
                                       times_table_p,
                                       free_when_done);

    return CoherentSummation::emission_tomography_method(gather,
                                                         receivers_coords,
                                                         environment_,
                                                         sources_receivers_times,
                                                         dt,
                                                         tensor_matrix,
                                                         receivers_block_size,
                                                         samples_block_size);
}

py_array_d CoherentSummationANN::emission_tomography_method(py_array_d gather,
                                                            double dt,
                                                            py_array_d receivers_coords,
                                                            std::ptrdiff_t receivers_block_size,
                                                            std::ptrdiff_t samples_block_size) {

    auto receivers_coords_info = receivers_coords.request();
    auto *receivers_coords_data = static_cast<double *>(receivers_coords_info.ptr);

    auto n_receivers = receivers_coords_info.shape[0];
    Array2D<double> receivers_coords2D;

    if (receivers_coords_info.ndim == 1) {
        receivers_coords2D = Array2D<double>(receivers_coords_data, n_receivers, 1,
                                             receivers_coords.strides()[0] / sizeof(double), 1);
    } else {
        receivers_coords2D = cohSumUtils::py_buffer_to_array2D<double>(receivers_coords.request());
    }

    std::cout << "Start times table calculating" << std::endl;
    auto times_table = time_arrival_nn_.get_times_to_points(receivers_coords2D);
    std::cout << "End times table calculating" << std::endl;

    auto n_points = environment_.shape(0);

    float *times_table_p = times_table.release();

    py::capsule free_when_done(times_table_p, [](void *f) {
        auto *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    py_array_f sources_receivers_times({n_points, n_receivers},
                                       {n_receivers * sizeof(float), sizeof(float)},
                                       times_table_p,
                                       free_when_done);

    return CoherentSummation::emission_tomography_method(gather,
                                                         sources_receivers_times,
                                                         dt,
                                                         receivers_block_size,
                                                         samples_block_size);
}

CoherentSummationANN::~CoherentSummationANN() noexcept = default;
