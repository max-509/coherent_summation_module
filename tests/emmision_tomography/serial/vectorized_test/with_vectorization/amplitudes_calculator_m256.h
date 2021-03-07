#ifndef _AMPLITUDES_CALCULATOR_M256_H
#define _AMPLITUDES_CALCULATOR_M256_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>

template <typename T>
class AmplitudesCalculatorM256 : public AmplitudesCalculatorBase<T, AmplitudesCalculatorM256<T>> {
public:
	AmplitudesCalculatorM256(const Array2D<T> &sources_coords,
						 	  const Array2D<T> &rec_coords,
						 	  const T *tensor_matrix,
						 	  Array2D<T> &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculatorBase<T, AmplitudesCalculatorM256<T>>;

private:
	const Array2D<T> &sources_coords_;
	const Array2D<T> &rec_coords_;
	const T *tensor_matrix_;
	Array2D<T> &amplitudes_;
    __m256d abs_mask_d = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    __m256 abs_mask_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256d d_epsilon_v = _mm256_set1_pd(std::numeric_limits<double>::epsilon());
    __m256 f_epsilon_v = _mm256_set1_ps(std::numeric_limits<float>::epsilon());

	void realize_calculate() {}

	inline __m256 vect_calc_norm(__m256 x, __m256 y, __m256 z) {
	    return _mm256_add_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), f_epsilon_v);
	}

	inline __m256d vect_calc_norm(__m256d x, __m256d y, __m256d z) {
	    return _mm256_add_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))), d_epsilon_v);
	}

};

template <>
void AmplitudesCalculatorM256<float>::realize_calculate() {
	std::ptrdiff_t n_rec = rec_coords_.get_y_dim();
    std::ptrdiff_t sources_count = sources_coords_.get_y_dim();
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);

    __m256 two_v = _mm256_set1_ps(2.0f);
    __m256 one_v = _mm256_set1_ps(1.0f);
    __m256 coord_vec[3];
    __m256 tensor_matrix_v[matrix_size] = {_mm256_set1_ps(tensor_matrix_[0]),
                                        _mm256_set1_ps(tensor_matrix_[1]),
                                        _mm256_set1_ps(tensor_matrix_[2]),
                                        _mm256_set1_ps(tensor_matrix_[3]),
                                        _mm256_set1_ps(tensor_matrix_[4]),
                                        _mm256_set1_ps(tensor_matrix_[5])
                                    };
    __m256 G_P_vect[matrix_size];

    __m256i vindex = _mm256_set_epi32(21, 18, 15, 12, 9, 6, 3, 0);
    
    for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
        __m256 sources_coords_v[3] = {_mm256_set1_ps(sources_coords_(i, 0)),
                                        _mm256_set1_ps(sources_coords_(i, 1)),
                                        _mm256_set1_ps(sources_coords_(i, 2))
                                    };
        for (std::ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                coord_vec[crd] = _mm256_sub_ps(_mm256_i32gather_ps(&rec_coords_(r_ind, crd), vindex, 4), sources_coords_v[crd]);
                // coord_vec[crd] = _mm256_sub_ps(_mm256_set_ps(rec_coords_(r_ind+7, crd), rec_coords_(r_ind+6, crd), rec_coords_(r_ind+5, crd), rec_coords_(r_ind+4, crd),
                //                                            rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)
                //                                            ), _mm256_set1_ps(sources_coords_(i, crd)));
            }

            __m256 rev_dist = _mm256_div_ps(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                coord_vec[crd] = _mm256_mul_ps(coord_vec[crd], rev_dist);
                G_P_vect[crd] = _mm256_mul_ps(_mm256_mul_ps(tensor_matrix_v[crd], _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[crd], coord_vec[crd]))), rev_dist);
            }

            G_P_vect[3] = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(tensor_matrix_v[3], _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[1], coord_vec[2]))), two_v), rev_dist);
            G_P_vect[4] = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(tensor_matrix_v[4], _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[0], coord_vec[2]))), two_v), rev_dist);
            G_P_vect[5] = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(tensor_matrix_v[5], _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[0], coord_vec[1]))), two_v), rev_dist);


            __m256 ampl_vect = _mm256_setzero_ps();
            for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                ampl_vect = _mm256_add_ps(ampl_vect, G_P_vect[m]);
            }

            _mm256_storeu_ps(&amplitudes_(i, r_ind), _mm256_div_ps(ampl_vect, _mm256_add_ps(_mm256_and_ps(ampl_vect, abs_mask_f), f_epsilon_v)));
        }
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

template <>
void AmplitudesCalculatorM256<double>::realize_calculate() {
	std::ptrdiff_t n_rec = rec_coords_.get_y_dim();
    std::ptrdiff_t sources_count = sources_coords_.get_y_dim();
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);

    __m256d two_v = _mm256_set1_pd(2.0);
    __m256d one_v = _mm256_set1_pd(1.0);
    __m256d coord_vec[3];
    __m256d tensor_matrix_v[matrix_size] = {_mm256_set1_pd(tensor_matrix_[0]),
                                        _mm256_set1_pd(tensor_matrix_[1]),
                                        _mm256_set1_pd(tensor_matrix_[2]),
                                        _mm256_set1_pd(tensor_matrix_[3]),
                                        _mm256_set1_pd(tensor_matrix_[4]),
                                        _mm256_set1_pd(tensor_matrix_[5])
                                    };
    __m256d G_P_vect[matrix_size];

    __m256i vindex = _mm256_set_epi64x(9, 6, 3, 0);

    for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
        __m256d sources_coords_v[3] = {_mm256_set1_pd(sources_coords_(i, 0)),
                                        _mm256_set1_pd(sources_coords_(i, 1)),
                                        _mm256_set1_pd(sources_coords_(i, 2))
                                    };
        for (std::ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                coord_vec[crd] = _mm256_sub_pd(_mm256_i64gather_pd(&rec_coords_(r_ind, crd), vindex, 8), sources_coords_v[crd]);
                // coord_vec[crd] = _mm256_sub_pd(_mm256_set_pd(rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), _mm256_set1_pd(sources_coords_(i, crd)));
            }

            __m256d rev_dist = _mm256_div_pd(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

            for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                coord_vec[crd] = _mm256_mul_pd(coord_vec[crd], rev_dist);
                G_P_vect[crd] = _mm256_mul_pd(_mm256_mul_pd(tensor_matrix_v[crd], _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[crd], coord_vec[crd]))), rev_dist);
            }

            G_P_vect[3] = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(tensor_matrix_v[3], _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[1], coord_vec[2]))), two_v), rev_dist);
            G_P_vect[4] = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(tensor_matrix_v[4], _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[0], coord_vec[2]))), two_v), rev_dist);
            G_P_vect[5] = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(tensor_matrix_v[5], _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[0], coord_vec[1]))), two_v), rev_dist);

            __m256d ampl_vect = _mm256_setzero_pd();
            for (std::ptrdiff_t m = 0; m < matrix_size; ++m) {
                ampl_vect = _mm256_add_pd(ampl_vect, G_P_vect[m]);
            }

            _mm256_storeu_pd(&amplitudes_(i, r_ind), _mm256_div_pd(ampl_vect, _mm256_add_pd(_mm256_and_pd(ampl_vect, abs_mask_d), d_epsilon_v)));                
        }
    }         

    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

#endif //_AMPLITUDES_CALCULATOR_M256_H
