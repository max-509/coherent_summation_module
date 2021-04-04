#ifndef _AMPLITUDES_CALCULATOR_H
#define _AMPLITUDES_CALCULATOR_H

#include <cstddef>
#include <cmath>
#include <limits>

template <typename T>
class AmplitudesCalculator {
public:

	AmplitudesCalculator(const Array2D<T> &sources_coords,
					 	  const T *tensor_matrix) :
		sources_coords_(sources_coords),
		tensor_matrix_(tensor_matrix)
	{ }

	void calculate(const Array2D<T> &rec_coords, Array2D<T> &amplitudes) {
		constexpr std::ptrdiff_t matrix_size = 6;
		std::ptrdiff_t n_sources = sources_coords_.get_y_dim();
		std::ptrdiff_t n_receivers = rec_coords.get_y_dim();

        T coord_vect[3];
        T G_P[matrix_size];
        constexpr T two_T = static_cast<T>(2.0);

        #pragma omp simd
        for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vect[crd] = rec_coords(i_r, crd)-sources_coords_(i_s, crd);
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
                for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                    ampl_tmp += (G_P[m])*tensor_matrix_[m];
                }
                ampl_tmp /= (std::fabs(ampl_tmp) + std::numeric_limits<T>::epsilon());

                amplitudes(i_s, i_r) = ampl_tmp;
            }
        }
	}

private:

    const Array2D<T> &sources_coords_;
	const T *tensor_matrix_;

	inline T calc_norm(const T *coord_vect, std::ptrdiff_t len) {
        T norm = 0.0;
        for (std::ptrdiff_t i_v = 0; i_v < len; ++i_v) {
            norm += coord_vect[i_v]*coord_vect[i_v];
        }
        return std::sqrt(norm) + std::numeric_limits<T>::epsilon();
	}
};

#endif //_AMPLITUDES_CALCULATOR_H