#include "CoherentSummation.h"

#include "array2D.h"
#include "array1D.h"
#include "emission_tomography_method.h"
#include "kirchhoff_migration.h"
#include "TimeArrivalBase.h"

#include <iostream>
#include <tuple>
#include <omp.h>

py_array_d
CoherentSummation::emission_tomography_method(py_array_d gather,
                                              py_array_f sources_receivers_times,
                                              double dt,
                                              std::ptrdiff_t receivers_block_size,
                                              std::ptrdiff_t samples_block_size) {

    auto gather_info = gather.request();
    auto sources_receivers_times_info = sources_receivers_times.request();

    if (gather_info.ndim != 2) {
        throw std::runtime_error("Error: Seismogram shape must be equal 2");
    }
    if (sources_receivers_times_info.ndim != 2) {
        throw std::runtime_error("Error: Travel times table's shape must be equal 2");
    }

    auto n_receivers = gather_info.shape[0], n_samples = gather_info.shape[1];
    auto n_points = sources_receivers_times_info.shape[0];

    if (n_receivers != sources_receivers_times_info.shape[1]) {
        throw std::runtime_error("Error: Number of receivers in seismogram and times table don't equal");
    }

    py::array_t<double> result = py::array_t<double>({static_cast<py::ssize_t>(n_points),
                                                      static_cast<py::ssize_t>(n_samples)});

    py::buffer_info result_info = result.request();

    auto *gather_data = static_cast<double *>(gather_info.ptr);
    auto *sources_receivers_times_data = static_cast<float *>(sources_receivers_times_info.ptr);
    auto *result_data = static_cast<double *>(result_info.ptr);

    Array2D<double> gather2D(gather_data, n_receivers, n_samples);
    Array2D<float> sources_receivers_times2D(sources_receivers_times_data, n_points, n_receivers);

    std::cout << "Start coherent summation:" << std::endl;
    emissionTomographyMethod(gather2D,
                             sources_receivers_times2D,
                             dt,
                             result_data,
                             receivers_block_size,
                             samples_block_size);
    std::cout << "End coherent summation:" << std::endl;

    return result;
}

py_array_d
CoherentSummation::emission_tomography_method(py_array_d gather,
                                              py_array_d receivers_coords,
                                              py_array_d sources_coords,
                                              py_array_f sources_receivers_times,
                                              double dt,
                                              py_array_d tensor_matrix,
                                              std::ptrdiff_t receivers_block_size,
                                              std::ptrdiff_t samples_block_size) {
    auto gather_info = gather.request();
    auto receivers_coords_info = receivers_coords.request();
    auto sources_coords_info = sources_coords.request();
    auto sources_receivers_times_info = sources_receivers_times.request();
    auto tensor_matrix_info = tensor_matrix.request();

    if (gather_info.ndim != 2) {
        throw std::runtime_error("Error: Seismogram shape must be equal 2");
    }
    if (sources_receivers_times_info.ndim != 2) {
        throw std::runtime_error("Error: Travel times table's shape must be equal 2");
    }
    if (tensor_matrix_info.ndim != 1 || tensor_matrix_info.shape[0] != 6) {
        throw std::runtime_error("Error: Bad shape of tensor moments matrix; "
                                 "shape must be equal 6 and must be had form: M11, M22, M33, M23, M13, M12");
    }

    auto n_receivers = gather_info.shape[0], n_samples = gather_info.shape[1];
    auto n_points = sources_receivers_times_info.shape[0];

    if (n_receivers != sources_receivers_times_info.shape[1]) {
        throw std::runtime_error("Error: Number of receivers in seismogram and times table don't equal");
    }
    if (n_receivers != receivers_coords_info.shape[0]) {
        throw std::runtime_error("Error: Number of receivers in seismogram and receivers coords don't equal");
    }
    if (n_points != sources_coords_info.shape[0]) {
        throw std::runtime_error("Error: Number of points in times table and sources coords don't equal");
    }
    if (sources_coords_info.ndim == 1 || sources_coords_info.shape[1] < 2) {
        throw std::runtime_error("Error: Number of coords in sources must be 2 or 3");
    }

    py::array_t<double> result = py::array_t<double>({static_cast<py::ssize_t>(n_points),
                                                      static_cast<py::ssize_t>(n_samples)});

    py::buffer_info result_info = result.request();

    auto *gather_data = static_cast<double *>(gather_info.ptr);
    auto *receivers_coords_data = static_cast<double *>(receivers_coords_info.ptr);
    auto *sources_coords_data = static_cast<double *>(sources_coords_info.ptr);
    auto *sources_receivers_times_data = static_cast<float *>(sources_receivers_times_info.ptr);
    auto *tensor_matrix_data = static_cast<double *>(tensor_matrix_info.ptr);
    auto *result_data = static_cast<double *>(result_info.ptr);

    double *receivers_coords_p, *sources_coords_p;
    std::vector<double> receivers_coords_vector, sources_coords_vector;

    if (receivers_coords_info.ndim == 1 || receivers_coords_info.shape[1] < 2) {
        receivers_coords_vector.resize(n_receivers * 3);
        #pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r];
            receivers_coords_vector[i_r * 3 + 1] = 0.0;
            receivers_coords_vector[i_r * 3 + 2] = 0.0;
        }
        receivers_coords_p = receivers_coords_vector.data();
    } else if (receivers_coords_info.ndim == 2) {
        receivers_coords_vector.resize(n_receivers * 3);
        #pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r * 2 + 0];
            receivers_coords_vector[i_r * 3 + 1] = receivers_coords_data[i_r * 2 + 1];
            receivers_coords_vector[i_r * 3 + 2] = 0.0;
        }
        receivers_coords_p = receivers_coords_vector.data();
    } else if (receivers_coords_info.ndim == 3) {
        receivers_coords_p = receivers_coords_data;
    }

    if (sources_coords_info.ndim == 2) {
        sources_coords_vector.resize(n_points * 3);
        #pragma omp parallel for
        for (auto i_p = 0; i_p < n_points; ++i_p) {
            sources_coords_vector[i_p * 3 + 0] = sources_coords_data[i_p * 2 + 0];
            sources_coords_vector[i_p * 3 + 1] = 0.0;
            sources_coords_vector[i_p * 3 + 2] = sources_coords_data[i_p * 2 + 1];
        }
        sources_coords_p = sources_coords_vector.data();
    } else if (sources_coords_info.ndim == 3) {
        sources_coords_p = sources_coords_data;
    }

    Array2D<double> gather2D(gather_data, n_receivers, n_samples);
    Array2D<double> receivers_coords2D(receivers_coords_p, n_receivers, 3);
    Array2D<double> sources_coords2D(sources_coords_p, n_points, 3);
    Array2D<float> sources_receivers_times2D(sources_receivers_times_data, n_points, n_receivers);

    std::cout << "Start coherent summation:" << std::endl;
    emissionTomographyMethod(gather2D,
                             receivers_coords2D,
                             sources_coords2D,
                             sources_receivers_times2D,
                             dt,
                             tensor_matrix_data,
                             result_data,
                             receivers_block_size,
                             samples_block_size);
    std::cout << "End coherent summation:" << std::endl;

    return result;
}

py_array_d
CoherentSummation::kirchhoff_migration_method(py_array_d gather,
                                                         py_array_f times_to_source,
                                                         py_array_f times_to_receivers,
                                                         double dt) {
    auto gather_info = gather.request();
    auto times_to_source_info = times_to_source.request();
    auto times_to_receivers_info = times_to_receivers.request();

    if (gather_info.ndim != 2) {
        throw std::runtime_error("Error: Seismogram shape must be equal 2");
    }
    if (times_to_source_info.ndim != 1) {
        throw std::runtime_error("Error: Times to source's table shape must be equal 1");
    }
    if (times_to_receivers_info.ndim != 2) {
        throw std::runtime_error("Error: Times to receivers table shape must be equal 2");
    }

    auto n_receivers = gather_info.shape[0], n_samples = gather_info.shape[1];
    auto n_points = times_to_receivers_info.shape[0];

    if (n_points != times_to_source_info.shape[0]) {
        throw std::runtime_error("Error: Number of points in source times table and receivers times table don't equal");
    }
    if (n_receivers != times_to_receivers_info.shape[1]) {
        throw std::runtime_error("Error: Number of receivers in seismogram and receivers times table don't equal");
    }

    py::array_t<double> result = py::array_t<double>(n_points);

    py::buffer_info result_info = result.request();

    auto *gather_data = static_cast<double *>(gather_info.ptr);
    auto *times_to_source_data = static_cast<float *>(times_to_source_info.ptr);
    auto *times_to_receivers_data = static_cast<float *>(times_to_receivers_info.ptr);
    auto *result_data = static_cast<double *>(result_info.ptr);

    Array2D<double> gather2D(gather_data, n_receivers, n_samples);
    Array1D<float> times_to_source1D(times_to_source_data, n_points);
    Array2D<float> times_to_receivers2D(times_to_receivers_data, n_points, n_receivers);

    std::cout << "Start coherent summation:" << std::endl;
    kirchhoffMigrationCHGByPoints(gather2D, times_to_source1D, times_to_receivers2D, dt, result_data);
    std::cout << "End coherent summation:" << std::endl;

    return result;
}

CoherentSummation::~CoherentSummation() noexcept = default;
