#ifndef _AMPLITUDES_CALCULATOR_M128_H
#define _AMPLITUDES_CALCULATOR_M128_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <x86intrin.h>
#include <functional>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <omp.h>

template <typename T>
class AmplitudesCalculatorM128 : public AmplitudesCalculatorBase<T, AmplitudesCalculatorM128<T>> {
public:
	AmplitudesCalculatorM128(const Array2D<T> &sources_coords,
						 	  const Array2D<T> &rec_coords,
						 	  const T *RESTRICT tensor_matrix,
						 	  Array2D<T> &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculatorBase<T, AmplitudesCalculatorM128<T>>;

private:
	const Array2D<T> &sources_coords_;
	const Array2D<T> &rec_coords_;
	const T *RESTRICT tensor_matrix_;
	Array2D<T> &amplitudes_;
    __m128d abs_mask_d = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    __m128 abs_mask_f = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128d d_epsilon_v = _mm_set1_pd(std::numeric_limits<double>::epsilon());
    __m128 f_epsilon_v = _mm_set1_ps(std::numeric_limits<float>::epsilon());

	void realize_calculate();

	inline __m128 vect_calc_norm(__m128 x, __m128 y, __m128 z) {
	    return _mm_add_ps(_mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z))), f_epsilon_v);
	}

	inline __m128d vect_calc_norm(__m128d x, __m128d y, __m128d z) {
	    return _mm_add_pd(_mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), _mm_mul_pd(z, z))), d_epsilon_v);
	}

};

template <>
void AmplitudesCalculatorM128<float>::realize_calculate() {
	std::ptrdiff_t n_rec = rec_coords_.get_y_dim();
    std::ptrdiff_t sources_count = sources_coords_.get_y_dim();
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t vector_dim = sizeof(__m128)/sizeof(float);

    __m128 two_v = _mm_set1_ps(2.0f);
    __m128 one_v = _mm_set1_ps(1.0f);
    __m128 tensor_matrix_v[matrix_size] = {_mm_set1_ps(tensor_matrix_[0]),
                                            _mm_set1_ps(tensor_matrix_[1]),
                                            _mm_set1_ps(tensor_matrix_[2]),
                                            _mm_set1_ps(tensor_matrix_[3]),
                                            _mm_set1_ps(tensor_matrix_[4]),
                                            _mm_set1_ps(tensor_matrix_[5])
                                        };

    #pragma omp parallel shared(tensor_matrix_v, two_v, one_v, vector_dim, sources_count, n_rec)
    {
        __m128 RESTRICT G_P_vect[matrix_size];
        __m128 RESTRICT coord_vec[3];

        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
            __m128 sources_coords_v[3] = {_mm_set1_ps(sources_coords_(i, 0)),
                                            _mm_set1_ps(sources_coords_(i, 1)),
                                            _mm_set1_ps(sources_coords_(i, 2))
                                        };
            for (std::ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_sub_ps(_mm_set_ps(rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), sources_coords_v[crd]);
                }

                __m128 rev_dist = _mm_div_ps(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_mul_ps(coord_vec[crd], rev_dist);
                    G_P_vect[crd] = _mm_mul_ps(_mm_mul_ps(tensor_matrix_v[crd], _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[crd], coord_vec[crd]))), rev_dist);
                }
                
                G_P_vect[3] = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(tensor_matrix_v[3], _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[1], coord_vec[2]))), two_v), rev_dist);
                G_P_vect[4] = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(tensor_matrix_v[4], _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[0], coord_vec[2]))), two_v), rev_dist);
                G_P_vect[5] = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(tensor_matrix_v[5], _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[0], coord_vec[1]))), two_v), rev_dist);

                __m128 ampl_vect = _mm_setzero_ps();
                for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                    ampl_vect = _mm_add_ps(ampl_vect, G_P_vect[m]);
                }

                _mm_storeu_ps(&amplitudes_(i, r_ind), _mm_div_ps(ampl_vect, _mm_add_ps(_mm_and_ps(ampl_vect, abs_mask_f), f_epsilon_v)));
            }
        }       
    }

    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

template <>
void AmplitudesCalculatorM128<double>::realize_calculate() {
	std::ptrdiff_t n_rec = rec_coords_.get_y_dim();
    std::ptrdiff_t sources_count = sources_coords_.get_y_dim();
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t vector_dim = sizeof(__m128d)/sizeof(double);

    __m128d two_v = _mm_set1_pd(2.0);
    __m128d one_v = _mm_set1_pd(1.0);
    __m128d tensor_matrix_v[matrix_size] = {_mm_set1_pd(tensor_matrix_[0]),
                                            _mm_set1_pd(tensor_matrix_[1]),
                                            _mm_set1_pd(tensor_matrix_[2]),
                                            _mm_set1_pd(tensor_matrix_[3]),
                                            _mm_set1_pd(tensor_matrix_[4]),
                                            _mm_set1_pd(tensor_matrix_[5])
                                        };
    

    #pragma omp parallel shared(tensor_matrix_v, two_v, one_v, vector_dim, sources_count, n_rec)
    {
        __m128d RESTRICT G_P_vect[matrix_size];
        __m128d RESTRICT coord_vec[3];

        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
            __m128d sources_coords_v[3] = {_mm_set1_pd(sources_coords_(i, 0)),
                                            _mm_set1_pd(sources_coords_(i, 1)),
                                            _mm_set1_pd(sources_coords_(i, 2))
                                        };
            for (std::ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_sub_pd(_mm_set_pd(rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), sources_coords_v[crd]);
                }

                __m128d rev_dist = _mm_div_pd(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_mul_pd(coord_vec[crd], rev_dist);
                    G_P_vect[crd] = _mm_mul_pd(_mm_mul_pd(tensor_matrix_v[crd], _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[crd], coord_vec[crd]))), rev_dist);
                }

                G_P_vect[3] = _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(tensor_matrix_v[3], _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[1], coord_vec[2]))), two_v), rev_dist);
                G_P_vect[4] = _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(tensor_matrix_v[4], _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[0], coord_vec[2]))), two_v), rev_dist);
                G_P_vect[5] = _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(tensor_matrix_v[5], _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[0], coord_vec[1]))), two_v), rev_dist);

                __m128d ampl_vect = _mm_setzero_pd();
                for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                    ampl_vect = _mm_add_pd(ampl_vect, G_P_vect[m]);
                }

                _mm_storeu_pd(&amplitudes_(i, r_ind), _mm_div_pd(ampl_vect, _mm_add_pd(_mm_and_pd(ampl_vect, abs_mask_d), d_epsilon_v)));
            }
        }
    }

    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

#endif //_AMPLITUDES_CALCULATOR_M128_H