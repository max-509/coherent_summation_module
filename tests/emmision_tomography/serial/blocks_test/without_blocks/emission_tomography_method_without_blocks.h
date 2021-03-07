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

template <typename T>
void emissionTomographyMethodWithoutBlocks(const Array2D<T> &gather, 
                                const Array2D<T> &receivers_coords,
                                const Array2D<T> &sources_coords,
                                const Array2D<T> &sources_receivers_times,
                                double dt,
                                const T *tensor_matrix,
                                T *result_data) {
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    T coord_vect[3];
    T G_P[matrix_size];
    constexpr T two_T = static_cast<T>(2.0);

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        T t_min_to_source = *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
        for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
            std::ptrdiff_t godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - t_min_to_source) * rev_dt);

            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                coord_vect[crd] = receivers_coords(i_r, crd) - sources_coords(i_s, crd);
            }

            T rev_dist = static_cast<T>(1.0) / calc_norm(coord_vect, 3);
            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                coord_vect[crd] *= rev_dist;    
                G_P[crd] = coord_vect[crd]*coord_vect[crd] * rev_dist;
            }

            G_P[0] *= coord_vect[2];
            G_P[1] *= coord_vect[2];
            G_P[2] *= coord_vect[2];
            G_P[3] = two_T*coord_vect[2]*coord_vect[1]*coord_vect[2] * rev_dist;
            G_P[4] = two_T*coord_vect[2]*coord_vect[0]*coord_vect[2] * rev_dist;
            G_P[5] = two_T*coord_vect[2]*coord_vect[0]*coord_vect[1] * rev_dist;

            T amplitude = 0.0;
            for (std::ptrdiff_t i_m = 0; i_m < matrix_size; ++i_m) {
                amplitude += G_P[i_m]*tensor_matrix[i_m];
            }

            amplitude /= (std::fabs(amplitude) + std::numeric_limits<T>::epsilon());

            for (std::ptrdiff_t i_t = 0; i_t < (n_samples - godograph_ind); ++i_t) {
                result_data[i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitude;
            }   
        }
    }
}

#endif //_EMISSION_TOMOGRAPHY_METHOD_WITHOUT_BLOCKS_H