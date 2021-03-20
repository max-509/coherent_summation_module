#ifndef _AMPLITUDES_CALCULATOR_BASE_H
#define _AMPLITUDES_CALCULATOR_BASE_H

#include "array2D.h"

#ifdef _MSC_VER
#define RESTRICT __restrict
#else 
#define RESTRICT __restrict__
#endif

#include <limits>
#include <cstddef>
#include <cmath>
#include <omp.h>

template <typename T, typename Realization>
class AmplitudesCalculatorBase {
public:

	void calculate(const Array2D<T> &rec_coords, Array2D<T> &amplitudes) {
		static_cast<Realization*>(this)->realize_calculate(rec_coords, amplitudes); 
	}

	friend Realization;

private:

	void non_vector_calculate_amplitudes(std::ptrdiff_t ind_first_rec, const Array2D<T> &sources_coords, const Array2D<T> &rec_coords, const T *RESTRICT tensor_matrix, Array2D<T> &amplitudes) {
	    std::ptrdiff_t n_rec = rec_coords.get_y_dim();
	    std::ptrdiff_t sources_count = sources_coords.get_y_dim();
	    constexpr std::ptrdiff_t matrix_size = 6;

	    constexpr T two_T = static_cast<T>(2.0);

        #pragma omp parallel
        {
            T RESTRICT coord_vect[3];
            T RESTRICT G_P[matrix_size];

            #pragma omp for simd schedule(static) collapse(2)
            for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
                for (std::ptrdiff_t r_ind = ind_first_rec; r_ind < n_rec; ++r_ind) {
                    for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                        coord_vect[crd] = rec_coords(r_ind, crd)-sources_coords(i, crd);
                    }

                    T rev_dist = static_cast<T>(1.0) / calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);
                    for (std::ptrdiff_t crd = 2; crd >= 0; --crd) {
                        coord_vect[crd] *= rev_dist;
                        G_P[crd] = coord_vect[2]*coord_vect[crd]*coord_vect[crd] * rev_dist;
                    }

                    T double_norm_coord_z = two_T*coord_vect[2]*rev_dist;

                    G_P[3] = double_norm_coord_z*coord_vect[1]*coord_vect[2];
                    G_P[4] = double_norm_coord_z*coord_vect[0]*coord_vect[2];
                    G_P[5] = double_norm_coord_z*coord_vect[0]*coord_vect[1];
                    

                    T ampl_tmp = 0.0;
                    for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                        ampl_tmp += (G_P[m])*tensor_matrix[m];
                    }
                    ampl_tmp /= (std::fabs(ampl_tmp) + std::numeric_limits<T>::epsilon());

                    amplitudes(i, r_ind) = ampl_tmp;
                }
            }
        }

	}

	#pragma omp declare simd
	inline T calc_norm(T x, T y, T z) {
	    return sqrt(x*x+y*y+z*z)+std::numeric_limits<T>::epsilon();
	}
};

#endif //_AMPLITUDES_CALCULATOR_BASE_H