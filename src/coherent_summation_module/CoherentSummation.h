#ifndef COHERENT_SUMMATION_H
#define COHERENT_SUMMATION_H

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <cmath>
#include <omp.h>

#include "utils.h"
#include "py_common.h"
#include "array2D.h"
#include "array1D.h"
#include "emission_tomography_method.h"
#include "kirchhoff_migration.h"


class CoherentSummation {
public:

    CoherentSummation() = default;

    virtual ~CoherentSummation() noexcept = default;

    template<typename T1, typename T2>
    py_array<T1>
    emission_tomography_method(py_array<T1> gather,
                               py_array<T2> sources_receivers_times,
                               double dt,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    template<typename T1, typename T2>
    py_array<T1>
    emission_tomography_method(py_array<T1> gather,
                               py_array<T1> receivers_coords,
                               py_array<T1> sources_coords,
                               py_array<T2> sources_receivers_times,
                               double dt,
                               py_array<T1> tensor_matrix,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    template<typename T1, typename T2>
    py_array<T1>
    kirchhoff_migration_method(py_array<T1> gather,
                               py_array<T2> times_to_source,
                               py_array<T2> times_to_receivers,
                               double dt,
                               std::ptrdiff_t p_block_size = 0);
};

template<typename T1, typename T2>
py_array<T1>
CoherentSummation::emission_tomography_method(py_array<T1> gather,
                                              py_array<T2> sources_receivers_times,
                                              double dt,
                                              std::ptrdiff_t receivers_block_size,
                                              std::ptrdiff_t samples_block_size) {

    std::cout << "BEFORE SLEEP" << std::endl;
    std::cout << "AFTER SLEEP" << std::endl;

    std::cout << "Sizeof T1 == " << sizeof(T1) << std::endl;
    std::cout << "Sizeof T2 == " << sizeof(T2) << std::endl;

    if (gather.dtype().is(py::dtype::of<float>())) {
        std::cerr << "Gather type is float" << std::endl;
    } else {
        std::cerr << "Gather type is double" << std::endl;
    }

    std::cerr << "Gather references counting: " << gather.ref_count() << std::endl;

    if (sources_receivers_times.dtype().is(py::dtype::of<float>())) {
        std::cerr << "Sources_receivers_times type is float" << std::endl;
    } else {
        std::cerr << "Sources_receivers_times type is double" << std::endl;
    }

    std::cerr << "Sources_receivers_times references counting: " << sources_receivers_times.ref_count() << std::endl;

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

    auto *gather_data = static_cast<T1 *>(gather_info.ptr);
    auto *sources_receivers_times_data = static_cast<T2 *>(sources_receivers_times_info.ptr);

    Array2D<T1> gather2D(gather_data, n_receivers, n_samples);
    Array2D<T2> sources_receivers_times2D(sources_receivers_times_data, n_points, n_receivers);

    py::gil_scoped_release release;
    std::cout << "Start coherent summation:" << std::endl;
    auto result_arr = emissionTomographyMethod(gather2D,
                                               sources_receivers_times2D,
                                               dt,
                                               receivers_block_size,
                                               samples_block_size);
    std::cout << "End coherent summation:" << std::endl;
    py::gil_scoped_acquire acquire;

    T1 *result_p = result_arr.get();

    py::capsule free_when_done(result_p, [](void *f) {
        auto *foo = reinterpret_cast<T1 *>(f);
        delete[] foo;
    });

    py_array<T1> result({(py::ssize_t) result_arr.get_y_dim(), (py::ssize_t) result_arr.get_x_dim()},
                        {result_arr.get_x_dim() * sizeof(T1), sizeof(T1)},
                        result_p,
                        free_when_done);

    return result;
}

template<typename T1, typename T2>
py_array<T1>
CoherentSummation::emission_tomography_method(py_array<T1> gather,
                                              py_array<T1> receivers_coords,
                                              py_array<T1> sources_coords,
                                              py_array<T2> sources_receivers_times,
                                              double dt,
                                              py_array<T1> tensor_matrix,
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

    auto *gather_data = static_cast<T1 *>(gather_info.ptr);
    auto *receivers_coords_data = static_cast<T1 *>(receivers_coords_info.ptr);
    auto *sources_coords_data = static_cast<T1 *>(sources_coords_info.ptr);
    auto *sources_receivers_times_data = static_cast<T2 *>(sources_receivers_times_info.ptr);
    auto *tensor_matrix_data = static_cast<T1 *>(tensor_matrix_info.ptr);
//    auto *result_data = static_cast<double *>(result_info.ptr);

    T1 *receivers_coords_p, *sources_coords_p;
    std::vector<T1> receivers_coords_vector, sources_coords_vector;
    auto receivers_coords_vector_dims = receivers_coords_info.shape[1];
    auto receivers_coords_ndim = receivers_coords_info.ndim;

    auto sources_coords_ndim = sources_coords_info.ndim;

    py::gil_scoped_release release;
    if (receivers_coords_ndim == 1 || receivers_coords_vector_dims < 2) {
        receivers_coords_vector.resize(n_receivers * 3);
#pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r];
            receivers_coords_vector[i_r * 3 + 1] = 0.0;
            receivers_coords_vector[i_r * 3 + 2] = 0.0;
        }
        receivers_coords_p = receivers_coords_vector.data();
    } else if (receivers_coords_ndim == 2) {
        receivers_coords_vector.resize(n_receivers * 3);
#pragma omp parallel for
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            receivers_coords_vector[i_r * 3 + 0] = receivers_coords_data[i_r * 2 + 0];
            receivers_coords_vector[i_r * 3 + 1] = receivers_coords_data[i_r * 2 + 1];
            receivers_coords_vector[i_r * 3 + 2] = 0.0;
        }
        receivers_coords_p = receivers_coords_vector.data();
    } else if (receivers_coords_ndim == 3) {
        receivers_coords_p = receivers_coords_data;
    }

    if (sources_coords_ndim == 2) {
        sources_coords_vector.resize(n_points * 3);
#pragma omp parallel for
        for (auto i_p = 0; i_p < n_points; ++i_p) {
            sources_coords_vector[i_p * 3 + 0] = sources_coords_data[i_p * 2 + 0];
            sources_coords_vector[i_p * 3 + 1] = 0.0;
            sources_coords_vector[i_p * 3 + 2] = sources_coords_data[i_p * 2 + 1];
        }
        sources_coords_p = sources_coords_vector.data();
    } else if (sources_coords_ndim == 3) {
        sources_coords_p = sources_coords_data;
    }

    Array2D<T1> gather2D(gather_data, n_receivers, n_samples);
    Array2D<T1> receivers_coords2D(receivers_coords_p, n_receivers, 3);
    Array2D<T1> sources_coords2D(sources_coords_p, n_points, 3);
    Array2D<T2> sources_receivers_times2D(sources_receivers_times_data, n_points, n_receivers);

    std::cout << "Start coherent summation:" << std::endl;
    auto result_arr = emissionTomographyMethod(gather2D,
                                               receivers_coords2D,
                                               sources_coords2D,
                                               sources_receivers_times2D,
                                               dt,
                                               tensor_matrix_data,
                                               receivers_block_size,
                                               samples_block_size);
    std::cout << "End coherent summation:" << std::endl;

    py::gil_scoped_acquire acquire;

    T1 *result_p = result_arr.get();

    py::capsule free_when_done(result_p, [](void *f) {
        auto *foo = reinterpret_cast<T1 *>(f);
        delete[] foo;
    });

    py_array<T1> result({(py::ssize_t) result_arr.get_y_dim(), (py::ssize_t) result_arr.get_x_dim()},
                        {result_arr.get_x_dim() * sizeof(T1), sizeof(T1)},
                        result_p,
                        free_when_done);

    return result;
}

template<typename T1, typename T2>
py_array<T1>
CoherentSummation::kirchhoff_migration_method(py_array<T1> gather,
                                              py_array<T2> times_to_source,
                                              py_array<T2> times_to_receivers,
                                              double dt,
                                              std::ptrdiff_t p_block_size) {
    if (p_block_size <= 0) {
        p_block_size = 80;
    }

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

    auto n_points_cbrt_decimal = static_cast<std::ptrdiff_t>(std::cbrt(n_points));
    auto n_points_cbrt_decimal_remainder =
            n_points - (n_points_cbrt_decimal * n_points_cbrt_decimal * n_points_cbrt_decimal);

    if (n_points != times_to_source_info.shape[0]) {
        throw std::runtime_error("Error: Number of points in source times table and receivers times table don't equal");
    }
    if (n_receivers != times_to_receivers_info.shape[1]) {
        throw std::runtime_error("Error: Number of receivers in seismogram and receivers times table don't equal");
    }

    py::array_t<T1> result;
    try {
        result = py::array_t<T1>(static_cast<py::ssize_t>(n_points));
    } catch (...) {
        throw std::runtime_error("Error: Cannot allocate memory for result of kirchhoff migration");
    }

    py::buffer_info result_info = result.request();

    auto *gather_data = static_cast<T1 *>(gather_info.ptr);
    auto *times_to_source_data = static_cast<T2 *>(times_to_source_info.ptr);
    auto *times_to_receivers_data = static_cast<T2 *>(times_to_receivers_info.ptr);
    auto *result_data = static_cast<T1 *>(result_info.ptr);

    Array2D<T1> gather2D(gather_data, n_receivers, n_samples);
    Array1D<T2> times_to_source1D(times_to_source_data, n_points);
    Array2D<T2> times_to_receivers2D(times_to_receivers_data, n_points, n_receivers);

    py::gil_scoped_release release;

    std::cout << "Start coherent summation:" << std::endl;
    kirchhoffMigrationCHG3D(gather2D, times_to_source1D, times_to_receivers2D, n_points_cbrt_decimal,
                            n_points_cbrt_decimal, n_points_cbrt_decimal + n_points_cbrt_decimal_remainder, dt,
                            result_data,
                            p_block_size);
    std::cout << "End coherent summation:" << std::endl;

    py::gil_scoped_acquire acquire;

    return result;
}


#endif //COHERENT_SUMMATION_H
