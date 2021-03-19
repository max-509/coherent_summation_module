#ifndef _AMPLITUDES_CALCULATOR_H
#define _AMPLITUDES_CALCULATOR_H

#ifdef _MSC_VER
#define RESTRICT __restrict
#else 
#define RESTRICT __restrict__
#endif

#include <cstddef>
#include <cmath>
#include <limits>
#include <omp.h>

template <typename T>
class AmplitudesCalculator {
public:

	AmplitudesCalculator(const Array2D<T> &sources_coords,
					 	  const Array2D<T> &receivers_coords,
					 	  const T *RESTRICT tensor_matrix,
					 	  Array2D<T> &amplitudes) : 
		sources_coords_(sources_coords),
		receivers_coords_(receivers_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes)
	{ }

	void calculate() {
		constexpr std::ptrdiff_t matrix_size = 6;
		std::ptrdiff_t n_sources = sources_coords_.get_y_dim();
		std::ptrdiff_t n_receivers = receivers_coords_.get_y_dim();

        constexpr T two_T = static_cast<T>(2.0);

        #pragma omp parallel
        {
            T RESTRICT coord_vect[3];
            T RESTRICT G_P[matrix_size];

            #pragma omp for simd schedule(static) collapse(2)
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
                for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

                    for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                        coord_vect[crd] = receivers_coords_(i_r, crd) - sources_coords_(i_s, crd);
                    }

                    T rev_dist = static_cast<T>(1.0) / calc_norm(coord_vect, 3);
                    for (std::ptrdiff_t crd = 2; crd >= 0; --crd) {
                        coord_vect[crd] *= rev_dist;    
                        G_P[crd] = coord_vect[2]*coord_vect[crd]*coord_vect[crd] * rev_dist;
                    }
                    
                    T double_norm_coord_z = two_T*coord_vect[2]*rev_dist;

                    G_P[3] = double_norm_coord_z*coord_vect[1]*coord_vect[2];
                    G_P[4] = double_norm_coord_z*coord_vect[0]*coord_vect[2];
                    G_P[5] = double_norm_coord_z*coord_vect[0]*coord_vect[1];

                    T ampl_tmp = 0.0;
                    for (std::ptrdiff_t i_m = 0; i_m < matrix_size; ++i_m) {
                        ampl_tmp += G_P[i_m]*tensor_matrix_[i_m];
                    }
                    ampl_tmp /= (std::fabs(ampl_tmp) + std::numeric_limits<T>::epsilon());

                    amplitudes_(i_s, i_r) = ampl_tmp;
                }
            }
        }
	}

private:

	const Array2D<T> &sources_coords_;
    const Array2D<T> &receivers_coords_;
    const T *RESTRICT tensor_matrix_;
    Array2D<T> &amplitudes_;	

    #pragma omp declare simd
	inline T calc_norm(const T *RESTRICT coord_vect, std::ptrdiff_t len) {
        T norm = 0.0;
        for (std::ptrdiff_t i_v = 0; i_v < len; ++i_v) {
            norm += coord_vect[i_v]*coord_vect[i_v];
        }
        return std::sqrt(norm) + std::numeric_limits<T>::epsilon();
	}
};

#endif //_AMPLITUDES_CALCULATOR_H