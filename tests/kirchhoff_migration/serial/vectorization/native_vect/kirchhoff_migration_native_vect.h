#ifndef _KIRCHHOFF_MIGRATION_NATIVE_VECT_H
#define _KIRCHHOFF_MIGRATION_NATIVE_VECT_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>

#include "kirchhoff_migration_by_points_impl.h"
#include "array2D.h"

template <typename T1, typename T2>
void kirchhoffMigrationCHG2DNativeVect(const Array2D<T1> &gather,
                                       const std::vector<T2> &times_to_source,
                                       const Array2D<T2> &times_to_receivers,
                                       std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                       double dt,
                                       T1 *result_data) {

    const double rev_dt = 1.0 / dt;

    const auto n_points = z_dim*x_dim;

    process_receivers_on_points(gather, times_to_source, times_to_receivers, 0, n_points, rev_dt, result_data);

//    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
//
//        auto i_p0 = i_z*x_dim;
//        auto i_pn = i_p0 + x_dim;
//        process_receivers_on_points(gather,times_to_source, times_to_receivers, i_p0, i_pn, rev_dt, result_data);
//
//    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3DNativeVect(const Array2D<T1> &gather,
                                       const std::vector<T2> &times_to_source,
                                       const Array2D<T2> &times_to_receivers,
                                       std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                       double dt,
                                       T1 *result_data) {

    const double rev_dt = 1.0 / dt;

    const auto n_points = z_dim*y_dim*x_dim;

    process_receivers_on_points(gather, times_to_source, times_to_receivers, 0, n_points, rev_dt, result_data);

//    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
//        for (std::ptrdiff_t i_y = 0; i_y < y_dim; ++i_y) {
//
//            const auto i_p0 = (i_z*y_dim + i_y)*x_dim;
//            const auto i_pn = i_p0 + x_dim;
//
//            process_receivers_on_points(gather,times_to_source, times_to_receivers, i_p0, i_pn, rev_dt, result_data);
//        }
//    }
}

#endif //_KIRCHHOFF_MIGRATION_NATIVE_VECT_H