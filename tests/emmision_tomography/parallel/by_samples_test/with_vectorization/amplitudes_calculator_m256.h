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
#include <omp.h>

template <typename T>
class AmplitudesCalculatorM256 : public AmplitudesCalculatorBase<T, AmplitudesCalculatorM256<T>> {
public:
	AmplitudesCalculatorM256(const Array2D<T> &sources_coords,
						 	  const T *RESTRICT tensor_matrix) : 
		sources_coords_(sources_coords),
		tensor_matrix_(tensor_matrix)
	{ }

	friend AmplitudesCalculatorBase<T, AmplitudesCalculatorM256<T>>;

private:
	const Array2D<T> &sources_coords_;
	const T *RESTRICT tensor_matrix_;
    __m256d abs_mask_d = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    __m256 abs_mask_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256d d_epsilon_v = _mm256_set1_pd(std::numeric_limits<double>::epsilon());
    __m256 f_epsilon_v = _mm256_set1_ps(std::numeric_limits<float>::epsilon());

	void realize_calculate(const Array2D<T> &rec_coords_, Array2D<T> &amplitudes_) {}

	inline __m256 vect_calc_norm(__m256 x, __m256 y, __m256 z) {
	    return _mm256_add_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), f_epsilon_v);
	}

	inline __m256d vect_calc_norm(__m256d x, __m256d y, __m256d z) {
	    return _mm256_add_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))), d_epsilon_v);
	}

};

template <>
void AmplitudesCalculatorM256<float>::realize_calculate(const Array2D<float> &rec_coords_, Array2D<float> &amplitudes_) {
	std::ptrdiff_t n_rec = rec_coords_.get_y_dim();
    std::ptrdiff_t sources_count = sources_coords_.get_y_dim();
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);

    static __m256 two_v = _mm256_set1_ps(2.0f);
    static __m256 RESTRICT tensor_matrix_v[matrix_size] = {_mm256_broadcast_ss(tensor_matrix_),
                                                            _mm256_broadcast_ss(tensor_matrix_ + 1),
                                                            _mm256_broadcast_ss(tensor_matrix_ + 2),
                                                            _mm256_broadcast_ss(tensor_matrix_ + 3),
                                                            _mm256_broadcast_ss(tensor_matrix_ + 4),
                                                            _mm256_broadcast_ss(tensor_matrix_ + 5)
                                                        };
    static __m256i vindex = _mm256_set_epi32(21, 18, 15, 12, 9, 6, 3, 0);

    #pragma omp parallel
    {
        __m256 RESTRICT coord_vec[3];

        #pragma omp for schedule(dynamic) collapse(2)
        for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
            for (std::ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_ps(_mm256_i32gather_ps(&rec_coords_(r_ind, crd), vindex, 4), _mm256_broadcast_ss(&sources_coords_(i, crd)));
                    // coord_vec[crd] = _mm256_sub_ps(_mm256_set_ps(rec_coords_(r_ind+7, crd), rec_coords_(r_ind+6, crd), rec_coords_(r_ind+5, crd), rec_coords_(r_ind+4, crd),
                    //                                            rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)
                    //                                            ), sources_coords_v[crd]);
                }

                __m256 rev_dist = _mm256_rcp_ps(vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                coord_vec[2] = _mm256_mul_ps(coord_vec[2], rev_dist);

                __m256 norm_coord_z = _mm256_mul_ps(coord_vec[2], rev_dist);

                __m256 ampl_vect = _mm256_mul_ps(tensor_matrix_v[2], _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[2], coord_vec[2])));

                coord_vec[1] = _mm256_mul_ps(coord_vec[1], rev_dist);

                #ifdef __FMA__
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[1], _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[1], coord_vec[1])), ampl_vect);
                coord_vec[0] = _mm256_mul_ps(coord_vec[0], rev_dist);
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[0], _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[0])), ampl_vect);

                __m256 double_norm_coord_z = _mm256_mul_ps(norm_coord_z, two_v);

                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[3], _mm256_mul_ps(double_norm_coord_z, _mm256_mul_ps(coord_vec[1], coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[4], _mm256_mul_ps(double_norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[5], _mm256_mul_ps(double_norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[1])), ampl_vect);
                #else
                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[1], _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[1], coord_vec[1]))));
                coord_vec[0] = _mm256_mul_ps(coord_vec[0], rev_dist);
                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[0], _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[0]))));

                __m256 double_norm_coord_z = _mm256_mul_ps(norm_coord_z, two_v);

                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[3], _mm256_mul_ps(double_norm_coord_z, _mm256_mul_ps(coord_vec[1], coord_vec[2]))));
                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[4], _mm256_mul_ps(double_norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[2]))));
                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[5], _mm256_mul_ps(double_norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[1]))));
                #endif

                _mm256_storeu_ps(&amplitudes_(i, r_ind), _mm256_div_ps(ampl_vect, _mm256_add_ps(_mm256_and_ps(ampl_vect, abs_mask_f), f_epsilon_v)));
            }
        }
    }
    
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

template <>
void AmplitudesCalculatorM256<double>::realize_calculate(const Array2D<double> &rec_coords_, Array2D<double> &amplitudes_) {
	std::ptrdiff_t n_rec = rec_coords_.get_y_dim();
    std::ptrdiff_t sources_count = sources_coords_.get_y_dim();
    constexpr std::ptrdiff_t matrix_size = 6;
    std::ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);

    static __m256d two_v = _mm256_set1_pd(2.0);
    static __m256d one_v = _mm256_set1_pd(1.0);
    static __m256d RESTRICT tensor_matrix_v[matrix_size] = {_mm256_broadcast_sd(tensor_matrix_),
                                                            _mm256_broadcast_sd(tensor_matrix_ + 1),
                                                            _mm256_broadcast_sd(tensor_matrix_ + 2),
                                                            _mm256_broadcast_sd(tensor_matrix_ + 3),
                                                            _mm256_broadcast_sd(tensor_matrix_ + 4),
                                                            _mm256_broadcast_sd(tensor_matrix_ + 5)
                                                        };
                                    
    static __m256i vindex = _mm256_set_epi64x(9, 6, 3, 0);

    #pragma omp parallel
    {
        __m256d RESTRICT coord_vec[3];

        #pragma omp for schedule(dynamic) collapse(2)
        for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
            for (std::ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_pd(_mm256_i64gather_pd(&rec_coords_(r_ind, crd), vindex, 8), _mm256_broadcast_sd(&sources_coords_(i, crd)));
                    // coord_vec[crd] = _mm256_sub_pd(_mm256_set_pd(rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), sources_coords_v[crd]);
                }

                __m256d rev_dist = _mm256_div_pd(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                coord_vec[2] = _mm256_mul_pd(coord_vec[2], rev_dist);

                __m256d norm_coord_z = _mm256_mul_pd(coord_vec[2], rev_dist);

                __m256d ampl_vect = _mm256_mul_pd(tensor_matrix_v[2], _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[2], coord_vec[2])));

                coord_vec[1] = _mm256_mul_pd(coord_vec[1], rev_dist);

                #ifdef __FMA__
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[1], _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[1], coord_vec[1])), ampl_vect);
                coord_vec[0] = _mm256_mul_pd(coord_vec[0], rev_dist);
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[0], _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[0])), ampl_vect);

                __m256d double_norm_coord_z = _mm256_mul_pd(norm_coord_z, two_v);

                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[3], _mm256_mul_pd(double_norm_coord_z, _mm256_mul_pd(coord_vec[1], coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[4], _mm256_mul_pd(double_norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[5], _mm256_mul_pd(double_norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[1])), ampl_vect);
                #else
                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[1], _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[1], coord_vec[1]))));
                coord_vec[0] = _mm256_mul_pd(coord_vec[0], rev_dist);
                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[0], _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[0]))));

                __m256d double_norm_coord_z = _mm256_mul_pd(norm_coord_z, two_v);

                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[3], _mm256_mul_pd(double_norm_coord_z, _mm256_mul_pd(coord_vec[1], coord_vec[2]))));
                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[4], _mm256_mul_pd(double_norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[2]))));
                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[5], _mm256_mul_pd(double_norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[1]))));
                #endif

                _mm256_storeu_pd(&amplitudes_(i, r_ind), _mm256_div_pd(ampl_vect, _mm256_add_pd(_mm256_and_pd(ampl_vect, abs_mask_d), d_epsilon_v)));                
            }
        }      
    }   

    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

#endif //_AMPLITUDES_CALCULATOR_M256_H
