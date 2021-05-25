#ifndef _KIRCHHOFF_MIGRATION_H
#define _KIRCHHOFF_MIGRATION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>

#include "kirchhoff_migration_by_points_impl.h"
#include "array2D.h"

template <typename T1, typename T2>
void kirchhoffMigrationCHGByPoints(const Array2D<T1> &gather,
                                       const std::vector<T2> &times_to_source,
                                       const Array2D<T2> &times_to_receivers,
                                       std::ptrdiff_t n_points,
                                       double dt,
                                       T1 *result_data) {

    const double rev_dt = 1.0 / dt;

    process_receivers_on_points(gather, times_to_source, times_to_receivers, 0, n_points, rev_dt, result_data);
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG2D(const Array2D<T1> &gather,
                                       const std::vector<T2> &times_to_source,
                                       const Array2D<T2> &times_to_receivers,
                                       std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                       double dt,
                                       T1 *result_data) {

    const double rev_dt = 1.0 / dt;

    const auto n_points = z_dim*x_dim;

    process_receivers_on_points(gather, times_to_source, times_to_receivers, 0, n_points, rev_dt, result_data);
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3D(const Array2D<T1> &gather,
                             const std::vector<T2> &times_to_source,
                             const Array2D<T2> &times_to_receivers,
                             std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                             double dt,
                             T1 *result_data) {

    const double rev_dt = 1.0 / dt;

    const auto n_points = z_dim * y_dim * x_dim;

    process_receivers_on_points(gather, times_to_source, times_to_receivers, 0, n_points, rev_dt, result_data);
}

#endif //_KIRCHHOFF_MIGRATION_H