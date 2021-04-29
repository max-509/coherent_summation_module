#ifndef _AMPLITUDES_CALCULATOR_M512_H
#define _AMPLITUDES_CALCULATOR_M512_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <immintrin.h>
#include <functional>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <memory>
#include <type_traits>

template <typename InputArrayType,
        typename std::enable_if<std::is_floating_point<typename InputArrayType::value_type>::value, bool>::type = true>
class AmplitudesCalculatorM512 : public AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorM512<InputArrayType>> {
public:
    using value_type = typename std::remove_const<typename InputArrayType::value_type>::type;
    using size_type = typename InputArrayType::size_type;

	AmplitudesCalculatorM512(const InputArrayType &sources_coords,
						 	  const value_type * tensor_matrix) :
		sources_coords_(sources_coords),
		tensor_matrix_(tensor_matrix)
	{ }

	friend AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorM512<InputArrayType>>;

private:
	const InputArrayType &sources_coords_;
	const value_type *tensor_matrix_;
    __m512d d_epsilon_v = _mm512_set1_pd(std::numeric_limits<double>::epsilon());
    __m512 f_epsilon_v = _mm512_set1_ps(std::numeric_limits<float>::epsilon());

    template <typename OutputArrayType,
            bool is_f = std::is_same<float, value_type>::value,
            bool is_d = std::is_same<double, value_type>::value>
	void realize_calculate(const InputArrayType &rec_coords_, OutputArrayType &amplitudes_) {
        if (is_f) {
            realize_calculate_f(rec_coords_, amplitudes_);
        } else if (is_d) {
            realize_calculate_d(rec_coords_, amplitudes_);
        }
    }

    template<typename OutputArrayType>
    void realize_calculate_f(const InputArrayType &rec_coords_, OutputArrayType &amplitudes_) {
        size_type n_rec = rec_coords_.get_y_dim();
        size_type sources_count = sources_coords_.get_y_dim();
        constexpr size_type matrix_size = 6;
        size_type vector_dim = sizeof(__m512) / sizeof(float);

        static __m512 two_v = _mm512_set1_ps(2.0f);
        static __m512 tensor_matrix_v[matrix_size] = {_mm512_set1_ps(tensor_matrix_[0]),
                                                      _mm512_set1_ps(tensor_matrix_[1]),
                                                      _mm512_set1_ps(tensor_matrix_[2]),
                                                      _mm512_set1_ps(tensor_matrix_[3]),
                                                      _mm512_set1_ps(tensor_matrix_[4]),
                                                      _mm512_set1_ps(tensor_matrix_[5])
        };
        static __m512i vindex = _mm512_set_epi32(45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);

        __m512 coord_vec[3];

#ifdef _MSC_VER
#pragma omp parallel for schedule(static) collapse(2)
#else
#pragma omp parallel for simd schedule(static) collapse(2)
#endif
        for (size_type i = 0; i < sources_count; ++i) {
            for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                for (size_type crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_ps(_mm512_i32gather_ps(vindex, &rec_coords_(r_ind, crd), 4),
                                                   _mm512_set1_ps(sources_coords_(i, crd)));
                    // coord_vec[crd] = _mm512_sub_ps(_mm512_set_ps(rec_coords_(r_ind+15, crd), rec_coords_(r_ind+14, crd), rec_coords_(r_ind+13, crd), rec_coords_(r_ind+12, crd),
                    //                                            rec_coords_(r_ind+11, crd), rec_coords_(r_ind+10, crd), rec_coords_(r_ind+9, crd), rec_coords_(r_ind+8, crd),
                    //                                            rec_coords_(r_ind+7, crd), rec_coords_(r_ind+6, crd), rec_coords_(r_ind+5, crd), rec_coords_(r_ind+4, crd),
                    //                                            rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), _mm512_set1_ps(sources_coords_(i, crd)));
                }

                __m512 rev_dist = _mm512_rcp14_ps(vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                coord_vec[2] = _mm512_mul_ps(coord_vec[2], rev_dist);

                __m512 norm_coord_z = _mm512_mul_ps(coord_vec[2], rev_dist);

                __m512 ampl_vect = _mm512_mul_ps(tensor_matrix_v[2],
                                                 _mm512_mul_ps(norm_coord_z,
                                                               _mm512_mul_ps(coord_vec[2], coord_vec[2])));

                coord_vec[1] = _mm512_mul_ps(coord_vec[1], rev_dist);

                ampl_vect = _mm512_fmadd_ps(tensor_matrix_v[1],
                                            _mm512_mul_ps(norm_coord_z, _mm512_mul_ps(coord_vec[1], coord_vec[1])),
                                            ampl_vect);
                coord_vec[0] = _mm512_mul_ps(coord_vec[0], rev_dist);
                ampl_vect = _mm512_fmadd_ps(tensor_matrix_v[0],
                                            _mm512_mul_ps(norm_coord_z, _mm512_mul_ps(coord_vec[0], coord_vec[0])),
                                            ampl_vect);

                __m512 double_norm_coord_z = _mm512_mul_ps(norm_coord_z, two_v);

                ampl_vect = _mm512_fmadd_ps(tensor_matrix_v[3],
                                            _mm512_mul_ps(double_norm_coord_z,
                                                          _mm512_mul_ps(coord_vec[1], coord_vec[2])),
                                            ampl_vect);
                ampl_vect = _mm512_fmadd_ps(tensor_matrix_v[4],
                                            _mm512_mul_ps(double_norm_coord_z,
                                                          _mm512_mul_ps(coord_vec[0], coord_vec[2])),
                                            ampl_vect);
                ampl_vect = _mm512_fmadd_ps(tensor_matrix_v[5],
                                            _mm512_mul_ps(double_norm_coord_z,
                                                          _mm512_mul_ps(coord_vec[0], coord_vec[1])),
                                            ampl_vect);

                _mm512_storeu_ps(&amplitudes_(i, r_ind),
                                 _mm512_div_ps(ampl_vect, _mm512_add_ps(_mm512_abs_ps(ampl_vect), f_epsilon_v)));
            }
        }

        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), sources_coords_, rec_coords_, tensor_matrix_,
                                        amplitudes_);
    }

    template<typename OutputArrayType>
    void realize_calculate_d(const InputArrayType &rec_coords_, OutputArrayType &amplitudes_) {
        size_type n_rec = rec_coords_.get_y_dim();
        size_type sources_count = sources_coords_.get_y_dim();
        constexpr size_type matrix_size = 6;
        size_type vector_dim = sizeof(__m512d) / sizeof(double);

        static __m512d two_v = _mm512_set1_pd(2.0);
        static __m512d tensor_matrix_v[matrix_size] = {_mm512_set1_pd(tensor_matrix_[0]),
                                                       _mm512_set1_pd(tensor_matrix_[1]),
                                                       _mm512_set1_pd(tensor_matrix_[2]),
                                                       _mm512_set1_pd(tensor_matrix_[3]),
                                                       _mm512_set1_pd(tensor_matrix_[4]),
                                                       _mm512_set1_pd(tensor_matrix_[5])
        };
        static __m512i vindex = _mm512_set_epi64(21, 18, 15, 12, 9, 6, 3, 0);

        __m512d coord_vec[3];

#ifdef _MSC_VER
#pragma omp parallel for schedule(static) collapse(2)
#else
#pragma omp parallel for simd schedule(static) collapse(2)
#endif
        for (size_type i = 0; i < sources_count; ++i) {
            for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                for (size_type crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_pd(_mm512_i64gather_pd(vindex, &rec_coords_(r_ind, crd), 8),
                                                   _mm512_set1_pd(sources_coords_(i, crd)));
                    // coord_vec[crd] = _mm512_sub_pd(_mm512_set_pd(rec_coords_(r_ind+7, crd), rec_coords_(r_ind+6, crd), rec_coords_(r_ind+5, crd), rec_coords_(r_ind+4, crd),
                    //                                            rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), _mm512_set1_pd(sources_coords_(i, crd)));
                }

                __m512d rev_dist = _mm512_rcp14_pd(vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                coord_vec[2] = _mm512_mul_pd(coord_vec[2], rev_dist);

                __m512d norm_coord_z = _mm512_mul_pd(coord_vec[2], rev_dist);

                __m512d ampl_vect = _mm512_mul_pd(tensor_matrix_v[2],
                                                  _mm512_mul_pd(norm_coord_z,
                                                                _mm512_mul_pd(coord_vec[2], coord_vec[2])));

                coord_vec[1] = _mm512_mul_pd(coord_vec[1], rev_dist);

                ampl_vect = _mm512_fmadd_pd(tensor_matrix_v[1],
                                            _mm512_mul_pd(norm_coord_z, _mm512_mul_pd(coord_vec[1], coord_vec[1])),
                                            ampl_vect);
                coord_vec[0] = _mm512_mul_pd(coord_vec[0], rev_dist);
                ampl_vect = _mm512_fmadd_pd(tensor_matrix_v[0],
                                            _mm512_mul_pd(norm_coord_z, _mm512_mul_pd(coord_vec[0], coord_vec[0])),
                                            ampl_vect);

                __m512d double_norm_coord_z = _mm512_mul_pd(norm_coord_z, two_v);

                ampl_vect = _mm512_fmadd_pd(tensor_matrix_v[3],
                                            _mm512_mul_pd(double_norm_coord_z,
                                                          _mm512_mul_pd(coord_vec[1], coord_vec[2])),
                                            ampl_vect);
                ampl_vect = _mm512_fmadd_pd(tensor_matrix_v[4],
                                            _mm512_mul_pd(double_norm_coord_z,
                                                          _mm512_mul_pd(coord_vec[0], coord_vec[2])),
                                            ampl_vect);
                ampl_vect = _mm512_fmadd_pd(tensor_matrix_v[5],
                                            _mm512_mul_pd(double_norm_coord_z,
                                                          _mm512_mul_pd(coord_vec[0], coord_vec[1])),
                                            ampl_vect);

                _mm512_storeu_pd(&amplitudes_(i, r_ind),
                                 _mm512_div_pd(ampl_vect, _mm512_add_pd(_mm512_abs_pd(ampl_vect), d_epsilon_v)));
            }
        }
        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), sources_coords_, rec_coords_, tensor_matrix_,
                                        amplitudes_);
    }

	inline __m512 vect_calc_norm(__m512 x, __m512 y, __m512 z) {
	    return _mm512_add_ps(_mm512_sqrt_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, x), _mm512_mul_ps(y, y)), _mm512_mul_ps(z, z))), f_epsilon_v);
	}

	inline __m512d vect_calc_norm(__m512d x, __m512d y, __m512d z) {
	    return _mm512_add_pd(_mm512_sqrt_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)), _mm512_mul_pd(z, z))), d_epsilon_v);
	}

};

#endif //_AMPLITUDES_CALCULATOR_M512_H