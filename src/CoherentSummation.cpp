#include "CoherentSummation.h"

#include "array2D.h"
#include "emission_tomography_method.h"
#include "TimeArrivalWrapperNN.h"
#include "TimeArrivalTimesTableArray.h"
#include "TimeArrivalTimesTableFile.h"

#include <tuple>
#include <iostream>

CoherentSummation::CoherentSummation(double z0, double z1, std::size_t z_dim,
                                     double x0, double x1, std::size_t x_dim,
                                     double y0, double y1, std::size_t y_dim) {
    double dz = (z1 - z0) / static_cast<double>(z_dim - 1);
    double dy;
    if (y_dim == 1) {
        dy = 0.0;
    } else {
        dy = (y1 - y0) / static_cast<double>(y_dim - 1);
    }
    double dx = (x1 - x0) / static_cast<double>(x_dim - 1);

    environment_ = py::array_t<double>(
            std::vector<py::ssize_t>{(py::ssize_t) z_dim, (py::ssize_t) y_dim, (py::ssize_t) x_dim, 3});

    auto raw_unchecked_mutable_environment = environment_.mutable_unchecked<4>();

    for (std::size_t i_z = 0; i_z < z_dim; ++i_z) {
        double z = z0 + static_cast<double>(i_z) * dz;
        for (std::size_t i_y = 0; i_y < y_dim; ++i_y) {
            double y = y0 + static_cast<double>(i_y) * dy;
            for (std::size_t i_x = 0; i_x < x_dim; ++i_x) {
                double x = x0 + static_cast<double>(i_x) * dx;
                raw_unchecked_mutable_environment(i_z, i_y, i_x, 0) = x;
                raw_unchecked_mutable_environment(i_z, i_y, i_x, 1) = y;
                raw_unchecked_mutable_environment(i_z, i_y, i_x, 2) = z;
            }
        }
    }
}

CoherentSummation::CoherentSummation(py::array_t<double, py::array::c_style | py::array::forcecast> environment) {
    if (environment.ndim() == 3) {
        environment_ = py::array_t<double>(std::vector<py::ssize_t>{environment.shape(0), 1, environment.shape(1), 3});
        auto raw_environment = environment_.mutable_unchecked<4>();
        for (auto i_z = 0; i_z < environment.shape(0); ++i_z) {
            for (auto i_x = 0; i_x < environment.shape(1); ++i_x) {
                raw_environment(i_z, 0, i_x, 0) = environment.at(i_z, i_x, 0);
                raw_environment(i_z, 0, i_x, 1) = 0.0;
                raw_environment(i_z, 0, i_x, 2) = environment.at(i_z, i_x, 1);
            }
        }
    } else if (environment.ndim() == 4) {
        environment_ = py::array_t<double>(
                std::vector<py::ssize_t>{environment.shape(0), environment.shape(1), environment.shape(2),
                                         environment.shape(3)});
        auto raw_environment = environment_.mutable_unchecked<4>();
        for (auto i_z = 0; i_z < environment.shape(0); ++i_z) {
            for (auto i_y = 0; i_y < environment.shape(1); ++i_y) {
                for (auto i_x = 0; i_x < environment.shape(2); ++i_x) {
                    raw_environment(i_z, i_y, i_x, 0) = environment.at(i_z, i_y, i_x, 0);
                    raw_environment(i_z, i_y, i_x, 1) = environment.at(i_z, i_y, i_x, 1);
                    raw_environment(i_z, i_y, i_x, 2) = environment.at(i_z, i_y, i_x, 2);
                }
            }
        }
    } else {
        throw std::runtime_error("Error: environment shape must be (z_dim, x_dim) or (z_dim, y_dim, x_dim)");
    }

}

CoherentSummation::CoherentSummation(py::array_t<double, py::array::c_style | py::array::forcecast> t_table,
                                     double z0, double z1, std::size_t z_dim,
                                     double x0, double x1, std::size_t x_dim,
                                     double y0, double y1, std::size_t y_dim) :
        CoherentSummation(z0, z1, z_dim, x0, x1, x_dim, y0, y1, y_dim) {
    p_time_arrival_ = std::unique_ptr<TimeArrivalBase>(
            new TimeArrivalTimesTableArray(t_table));
}

CoherentSummation::CoherentSummation(py::array_t<double, py::array::c_style | py::array::forcecast> t_table,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> environment) :
        CoherentSummation(environment) {
    p_time_arrival_ = std::unique_ptr<TimeArrivalBase>(
            new TimeArrivalTimesTableArray(t_table));
}

CoherentSummation::CoherentSummation(const std::string &times_table_filename,
                                     double z0, double z1, std::size_t z_dim,
                                     double x0, double x1, std::size_t x_dim,
                                     double y0, double y1, std::size_t y_dim) :
        CoherentSummation(z0, z1, z_dim, x0, x1, x_dim, y0, y1, y_dim) {
    p_time_arrival_ = std::make_unique<TimeArrivalTimesTableFile>(times_table_filename, environment_.size() / 3);
}

CoherentSummation::CoherentSummation(const std::string &times_table_filename,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> environment) :
        CoherentSummation(environment) {
    p_time_arrival_ = std::make_unique<TimeArrivalTimesTableFile>(times_table_filename, environment_.size() / 3);
}

CoherentSummation::CoherentSummation(const std::string &path_NN, NN_Type nn_type,
                                     std::vector<std::pair<std::string, int>> &input_ops,
                                     std::vector<std::pair<std::string, int>> &output_ops,
                                     double z0, double z1, std::size_t z_dim,
                                     double x0, double x1, std::size_t x_dim,
                                     double y0, double y1, std::size_t y_dim) :
        CoherentSummation(z0, z1, z_dim, x0, x1, x_dim, y0, y1, y_dim) {
    switch (nn_type) {
        case NN_Type::FROZEN:
            p_time_arrival_ = std::unique_ptr<TimeArrivalBase>(
                    new TimeArrivalWrapperNN(environment_, path_NN, input_ops, output_ops));
            break;
        case NN_Type::MODEL:
            const char *tags = "serve";
            p_time_arrival_ = std::unique_ptr<TimeArrivalBase>(
                    new TimeArrivalWrapperNN(environment_, path_NN, &tags, 1, input_ops, output_ops));
    }


}

CoherentSummation::CoherentSummation(const std::string &path_NN, NN_Type nn_type,
                                     std::vector<std::pair<std::string, int>> &input_ops,
                                     std::vector<std::pair<std::string, int>> &output_ops,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> environment) :
        CoherentSummation(environment) {
    switch (nn_type) {
        case NN_Type::FROZEN:
            p_time_arrival_ = std::unique_ptr<TimeArrivalBase>(
                    new TimeArrivalWrapperNN(environment_, path_NN, input_ops, output_ops));
            break;
        case NN_Type::MODEL:
            const char *tags = "serve";
            p_time_arrival_ = std::unique_ptr<TimeArrivalBase>(
                    new TimeArrivalWrapperNN(environment_, path_NN, &tags, 1, input_ops, output_ops));
    }
}

py::array_t<double, py::array::c_style | py::array::forcecast>
CoherentSummation::emission_tomography_method(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &gather,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &receivers_coords,
        double dt,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &tensor_matrix) {

    py::buffer_info gather_info = gather.request(),
            receivers_coords_info = receivers_coords.request(),
            environment_info = environment_.request(),
            tensor_matrix_info = tensor_matrix.request();

    if (gather_info.ndim != 2) {
        throw std::runtime_error("Error: Seismogram shape must be equal 2");
    }
    std::ptrdiff_t n_receivers = gather_info.shape[0], n_samples = gather_info.shape[1];
    if (n_receivers != receivers_coords_info.shape[0]) {
        throw std::runtime_error("Error: Number of receivers in seismogram and coords don't equal");
    }
    if (tensor_matrix_info.ndim != 1 || tensor_matrix_info.shape[0] != 6) {
        throw std::runtime_error("Error: Bad shape of tensor moments matrix; "
                                 "shape must be equal 6 and must be had form: M11, M22, M33, M23, M13, M12");
    }

    auto n_points = environment_info.size / 3;

    py::array_t<double> result;
    if (environment_.shape(1) == 1) {
        result = py::array_t<double>({static_cast<py::ssize_t>(environment_.shape(0)),
                                      static_cast<py::ssize_t>(environment_.shape(2)),
                                      static_cast<py::ssize_t>(n_samples)});
    } else {
        result = py::array_t<double>({static_cast<py::ssize_t>(environment_.shape(0)),
                                      static_cast<py::ssize_t>(environment_.shape(1)),
                                      static_cast<py::ssize_t>(environment_.shape(2)),
                                      static_cast<py::ssize_t>(n_samples)});
    }

    py::buffer_info result_info = result.request();

    auto *gather_data = static_cast<double *>(gather_info.ptr);
    auto *receivers_coords_data = static_cast<double *>(receivers_coords_info.ptr);
    auto *environment_data = static_cast<double *>(environment_info.ptr);
    auto *result_data = static_cast<double *>(result_info.ptr);

    std::vector<double> receivers_coords_vector(n_receivers * 3);

    if (receivers_coords_info.ndim == 1 || receivers_coords_info.shape[1] == 1) {
        #pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r];
            receivers_coords_vector[i_r * 3 + 1] = 0.0;
            receivers_coords_vector[i_r * 3 + 2] = 0.0;
        }
    } else if (receivers_coords_info.shape[1] == 2) {
        #pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r * 2 + 0];
            receivers_coords_vector[i_r * 3 + 1] = receivers_coords_data[i_r * 2 + 1];
            receivers_coords_vector[i_r * 3 + 2] = 0.0;
        }
    } else if (receivers_coords_info.shape[1] == 3) {
        #pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r * 3 + 0];
            receivers_coords_vector[i_r * 3 + 1] = receivers_coords_data[i_r * 3 + 1];
            receivers_coords_vector[i_r * 3 + 2] = receivers_coords_data[i_r * 3 + 2];
        }
    }


    std::cerr << "Start calculating times arrivals:" << std::endl;
    auto p_times_to_receivers = p_time_arrival_->get_times_to_receivers(receivers_coords_vector, true);
    std::cerr << "End calculating times arrivals:" << std::endl;

    auto *tensor_matrix_data = static_cast<double *>(tensor_matrix_info.ptr);
    Array2D<double> gather2D(gather_data, n_receivers, n_samples);
    Array2D<double> receivers_coords2D(receivers_coords_vector.data(), n_receivers, 3);
    Array2D<double> sources_coords2D(environment_data, n_points, 3);
    Array2D<float> times_to_receivers2D(p_times_to_receivers.get(), n_points, n_receivers);

    std::cerr << "Start coherent summation:" << std::endl;
    if (cohSumUtils::is_tensor_matrix_diag(tensor_matrix)) {
        emissionTomographyMethod(gather2D,
                                 receivers_coords2D,
                                 sources_coords2D,
                                 times_to_receivers2D,
                                 dt,
                                 result_data);
    } else {
        emissionTomographyMethod(gather2D,
                                 receivers_coords2D,
                                 sources_coords2D,
                                 times_to_receivers2D,
                                 dt,
                                 tensor_matrix_data,
                                 result_data);
    }

    std::cerr << "End coherent summation:" << std::endl;

    return result;
}
