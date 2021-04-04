#ifndef _EMISSION_TOMOGRAPHY_METHOD_WITHOUT_BLOCKS_H
#define _EMISSION_TOMOGRAPHY_METHOD_WITHOUT_BLOCKS_H

#include "array2D.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstddef>

template <typename T>
inline T calc_norm(const T *coord_vect, std::ptrdiff_t len) {
    T norm = 0.0;
    for (std::ptrdiff_t i_v = 0; i_v < len; ++i_v) {
        norm += coord_vect[i_v]*coord_vect[i_v];
    }
    return std::sqrt(norm) + std::numeric_limits<T>::epsilon();
}

template <typename T1, typename T2>
void emissionTomographyMethodWithoutBlocks(const Array2D<T1> &gather,
                                const Array2D<T1> &receivers_coords,
                                const Array2D<T1> &sources_coords,
                                const Array2D<T2> &sources_receivers_times,
                                double dt,
                                const T1 *tensor_matrix,
                                T1 *result_data) {
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    T1 coord_vect[3];
    T1 G_P[matrix_size];
    constexpr T1 two_T = static_cast<T1>(2.0);
    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        T2 t_min_to_source = *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
        for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
            std::ptrdiff_t godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - t_min_to_source) * rev_dt);

            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vect[crd] = receivers_coords(i_r, crd)-sources_coords(i_s, crd);
                }

                T1 rev_dist = static_cast<T1>(1.0) / calc_norm(coord_vect, 3);
                for (std::ptrdiff_t crd = 2; crd >= 0; --crd) {
                    coord_vect[crd] *= rev_dist;
                    G_P[crd] = coord_vect[2]*coord_vect[crd]*coord_vect[crd] * rev_dist;
                }

                T1 double_norm_coord_z = two_T*coord_vect[2]*rev_dist;

                G_P[3] = double_norm_coord_z*coord_vect[1]*coord_vect[2];
                G_P[4] = double_norm_coord_z*coord_vect[0]*coord_vect[2];
                G_P[5] = double_norm_coord_z*coord_vect[0]*coord_vect[1];


                T1 amplitude = 0.0;
                for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                    amplitude += (G_P[m])*tensor_matrix[m];
                }
                amplitude /= (std::fabs(amplitude) + std::numeric_limits<T1>::epsilon());

            for (std::ptrdiff_t i_t = 0; i_t < (n_samples - godograph_ind); ++i_t) {
                result_data[i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitude;
            }   
        }
    }
}

#endif //_EMISSION_TOMOGRAPHY_METHOD_WITHOUT_BLOCKS_H