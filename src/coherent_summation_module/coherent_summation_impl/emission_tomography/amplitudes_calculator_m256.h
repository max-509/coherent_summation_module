#ifndef _AMPLITUDES_CALCULATOR_M256_H
#define _AMPLITUDES_CALCULATOR_M256_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <immintrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

template<typename InputArrayType,
        typename std::enable_if<std::is_floating_point<typename InputArrayType::value_type>::value, bool>::type = true>
class AmplitudesCalculatorM256
        : public AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorM256<InputArrayType>> {
public:
    using value_type = typename std::remove_const<typename InputArrayType::value_type>::type;
    using size_type = typename InputArrayType::size_type;

    AmplitudesCalculatorM256(const InputArrayType &sources_coords,
                             const value_type *tensor_matrix) :
            sources_coords_(sources_coords),
            tensor_matrix_(tensor_matrix) {}

    friend AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorM256<InputArrayType>>;

private:
    const InputArrayType &sources_coords_;
    const value_type *tensor_matrix_;
    __m256d abs_mask_d = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    __m256 abs_mask_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256d d_epsilon_v = _mm256_set1_pd(std::numeric_limits<double>::epsilon());
    __m256 f_epsilon_v = _mm256_set1_ps(std::numeric_limits<float>::epsilon());

    template<typename OutputArrayType>
    void realize_calculate(const InputArrayType &rec_coords_, OutputArrayType &amplitudes_) {
        realize_calculate_impl(rec_coords_, amplitudes_);
    }

    template<template<class> class OutputArrayType>
    void realize_calculate_impl(const InputArrayType &rec_coords_, OutputArrayType<float> &amplitudes_) {
        static_assert(std::is_same<float, value_type>::value,
                      "Error: In amplitudes calculator SIMD implementation types input and output arrays must be equal");
        size_type n_rec = rec_coords_.get_y_dim();
        size_type sources_count = sources_coords_.get_y_dim();
        constexpr size_type matrix_size = 6;
        constexpr size_type vector_dim = sizeof(__m256) / sizeof(float);

        static const __m256 two_v = _mm256_set1_ps(2.0f);
        __m256 tensor_matrix_v[matrix_size] = {_mm256_broadcast_ss(tensor_matrix_),
                                               _mm256_broadcast_ss(tensor_matrix_ + 1),
                                               _mm256_broadcast_ss(tensor_matrix_ + 2),
                                               _mm256_broadcast_ss(tensor_matrix_ + 3),
                                               _mm256_broadcast_ss(tensor_matrix_ + 4),
                                               _mm256_broadcast_ss(tensor_matrix_ + 5)
        };
        static const __m256i vindex = _mm256_set_epi32(21, 18, 15, 12, 9, 6, 3, 0);

        __m256 coord_vec[3];

#ifdef _MSC_VER
#pragma omp parallel for schedule(static) collapse(2)
#else
#pragma omp parallel for simd schedule(static) collapse(2)
#endif
        for (size_type i = 0; i < sources_count; ++i) {
            for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                for (size_type crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_ps(_mm256_i32gather_ps(&rec_coords_(r_ind, crd), vindex, 4),
                                                   _mm256_broadcast_ss(&sources_coords_(i, crd)));
                    // coord_vec[crd] = _mm256_sub_ps(_mm256_set_ps(rec_coords_(r_ind+7, crd), rec_coords_(r_ind+6, crd), rec_coords_(r_ind+5, crd), rec_coords_(r_ind+4, crd),
                    //                                            rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)
                    //                                            ), sources_coords_v[crd]);
                }

                __m256 rev_dist = _mm256_rcp_ps(vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                coord_vec[2] = _mm256_mul_ps(coord_vec[2], rev_dist);

                __m256 norm_coord_z = _mm256_mul_ps(coord_vec[2], rev_dist);

                __m256 ampl_vect = _mm256_mul_ps(tensor_matrix_v[2],
                                                 _mm256_mul_ps(norm_coord_z,
                                                               _mm256_mul_ps(coord_vec[2], coord_vec[2])));

                coord_vec[1] = _mm256_mul_ps(coord_vec[1], rev_dist);

#ifdef __FMA__
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[1],
                                            _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[1], coord_vec[1])),
                                            ampl_vect);
                coord_vec[0] = _mm256_mul_ps(coord_vec[0], rev_dist);
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[0],
                                            _mm256_mul_ps(norm_coord_z, _mm256_mul_ps(coord_vec[0], coord_vec[0])),
                                            ampl_vect);

                __m256 double_norm_coord_z = _mm256_mul_ps(norm_coord_z, two_v);

                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[3], _mm256_mul_ps(double_norm_coord_z,
                                                                              _mm256_mul_ps(coord_vec[1],
                                                                                            coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[4], _mm256_mul_ps(double_norm_coord_z,
                                                                              _mm256_mul_ps(coord_vec[0],
                                                                                            coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_ps(tensor_matrix_v[5], _mm256_mul_ps(double_norm_coord_z,
                                                                              _mm256_mul_ps(coord_vec[0],
                                                                                            coord_vec[1])), ampl_vect);
#else
                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[1], _mm256_mul_ps(norm_coord_z,
                                                                                                     _mm256_mul_ps(
                                                                                                             coord_vec[1],
                                                                                                             coord_vec[1]))));
                coord_vec[0] = _mm256_mul_ps(coord_vec[0], rev_dist);
                ampl_vect = _mm256_add_ps(ampl_vect, _mm256_mul_ps(tensor_matrix_v[0], _mm256_mul_ps(norm_coord_z,
                                                                                                     _mm256_mul_ps(
                                                                                                             coord_vec[0],
                                                                                                             coord_vec[0]))));

                __m256 double_norm_coord_z = _mm256_mul_ps(norm_coord_z, two_v);

                ampl_vect = _mm256_add_ps(ampl_vect,
                                          _mm256_mul_ps(tensor_matrix_v[3], _mm256_mul_ps(double_norm_coord_z,
                                                                                          _mm256_mul_ps(
                                                                                                  coord_vec[1],
                                                                                                  coord_vec[2]))));
                ampl_vect = _mm256_add_ps(ampl_vect,
                                          _mm256_mul_ps(tensor_matrix_v[4], _mm256_mul_ps(double_norm_coord_z,
                                                                                          _mm256_mul_ps(
                                                                                                  coord_vec[0],
                                                                                                  coord_vec[2]))));
                ampl_vect = _mm256_add_ps(ampl_vect,
                                          _mm256_mul_ps(tensor_matrix_v[5], _mm256_mul_ps(double_norm_coord_z,
                                                                                          _mm256_mul_ps(
                                                                                                  coord_vec[0],
                                                                                                  coord_vec[1]))));
#endif

                _mm256_storeu_ps(&amplitudes_(i, r_ind), _mm256_div_ps(ampl_vect,
                                                                       _mm256_add_ps(
                                                                               _mm256_and_ps(ampl_vect, abs_mask_f),
                                                                               f_epsilon_v)));
            }
        }

        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), sources_coords_, rec_coords_,
                                              tensor_matrix_,
                                              amplitudes_);
    }

    template<template<class> class OutputArrayType>
    void realize_calculate_impl(const InputArrayType &rec_coords_, OutputArrayType<double> &amplitudes_) {
        static_assert(std::is_same<double, value_type>::value,
                      "Error: In amplitudes calculator SIMD implementation types input and output arrays must be equal");
        size_type n_rec = rec_coords_.get_y_dim();
        size_type sources_count = sources_coords_.get_y_dim();
        constexpr size_type matrix_size = 6;
        constexpr size_type vector_dim = sizeof(__m256d) / sizeof(double);

        static const __m256d two_v = _mm256_set1_pd(2.0);
        static const __m256d one_v = _mm256_set1_pd(1.0);
        __m256d tensor_matrix_v[matrix_size] = {_mm256_broadcast_sd(tensor_matrix_),
                                                _mm256_broadcast_sd(tensor_matrix_ + 1),
                                                _mm256_broadcast_sd(tensor_matrix_ + 2),
                                                _mm256_broadcast_sd(tensor_matrix_ + 3),
                                                _mm256_broadcast_sd(tensor_matrix_ + 4),
                                                _mm256_broadcast_sd(tensor_matrix_ + 5)
        };

        static const __m256i vindex = _mm256_set_epi64x(9, 6, 3, 0);

        __m256d coord_vec[3];

#ifdef _MSC_VER
#pragma omp parallel for schedule(static) collapse(2)
#else
#pragma omp parallel for simd schedule(static) collapse(2)
#endif
        for (size_type i = 0; i < sources_count; ++i) {
            for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                for (size_type crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_pd(_mm256_i64gather_pd(&rec_coords_(r_ind, crd), vindex, 8),
                                                   _mm256_broadcast_sd(&sources_coords_(i, crd)));
                    // coord_vec[crd] = _mm256_sub_pd(_mm256_set_pd(rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), sources_coords_v[crd]);
                }

                __m256d rev_dist = _mm256_div_pd(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                coord_vec[2] = _mm256_mul_pd(coord_vec[2], rev_dist);

                __m256d norm_coord_z = _mm256_mul_pd(coord_vec[2], rev_dist);

                __m256d ampl_vect = _mm256_mul_pd(tensor_matrix_v[2],
                                                  _mm256_mul_pd(norm_coord_z,
                                                                _mm256_mul_pd(coord_vec[2], coord_vec[2])));

                coord_vec[1] = _mm256_mul_pd(coord_vec[1], rev_dist);

#ifdef __FMA__
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[1],
                                            _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[1], coord_vec[1])),
                                            ampl_vect);
                coord_vec[0] = _mm256_mul_pd(coord_vec[0], rev_dist);
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[0],
                                            _mm256_mul_pd(norm_coord_z, _mm256_mul_pd(coord_vec[0], coord_vec[0])),
                                            ampl_vect);

                __m256d double_norm_coord_z = _mm256_mul_pd(norm_coord_z, two_v);

                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[3], _mm256_mul_pd(double_norm_coord_z,
                                                                              _mm256_mul_pd(coord_vec[1],
                                                                                            coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[4], _mm256_mul_pd(double_norm_coord_z,
                                                                              _mm256_mul_pd(coord_vec[0],
                                                                                            coord_vec[2])), ampl_vect);
                ampl_vect = _mm256_fmadd_pd(tensor_matrix_v[5], _mm256_mul_pd(double_norm_coord_z,
                                                                              _mm256_mul_pd(coord_vec[0],
                                                                                            coord_vec[1])), ampl_vect);
#else
                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[1], _mm256_mul_pd(norm_coord_z,
                                                                                                     _mm256_mul_pd(
                                                                                                             coord_vec[1],
                                                                                                             coord_vec[1]))));
                coord_vec[0] = _mm256_mul_pd(coord_vec[0], rev_dist);
                ampl_vect = _mm256_add_pd(ampl_vect, _mm256_mul_pd(tensor_matrix_v[0], _mm256_mul_pd(norm_coord_z,
                                                                                                     _mm256_mul_pd(
                                                                                                             coord_vec[0],
                                                                                                             coord_vec[0]))));

                __m256d double_norm_coord_z = _mm256_mul_pd(norm_coord_z, two_v);

                ampl_vect = _mm256_add_pd(ampl_vect,
                                          _mm256_mul_pd(tensor_matrix_v[3], _mm256_mul_pd(double_norm_coord_z,
                                                                                          _mm256_mul_pd(
                                                                                                  coord_vec[1],
                                                                                                  coord_vec[2]))));
                ampl_vect = _mm256_add_pd(ampl_vect,
                                          _mm256_mul_pd(tensor_matrix_v[4], _mm256_mul_pd(double_norm_coord_z,
                                                                                          _mm256_mul_pd(
                                                                                                  coord_vec[0],
                                                                                                  coord_vec[2]))));
                ampl_vect = _mm256_add_pd(ampl_vect,
                                          _mm256_mul_pd(tensor_matrix_v[5], _mm256_mul_pd(double_norm_coord_z,
                                                                                          _mm256_mul_pd(
                                                                                                  coord_vec[0],
                                                                                                  coord_vec[1]))));
#endif

                _mm256_storeu_pd(&amplitudes_(i, r_ind), _mm256_div_pd(ampl_vect,
                                                                       _mm256_add_pd(
                                                                               _mm256_and_pd(ampl_vect, abs_mask_d),
                                                                               d_epsilon_v)));
            }
        }

        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), sources_coords_, rec_coords_,
                                              tensor_matrix_,
                                              amplitudes_);
    }

    inline __m256 vect_calc_norm(__m256 x, __m256 y, __m256 z) {
        return _mm256_add_ps(_mm256_sqrt_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))),
                             f_epsilon_v);
    }

    inline __m256d vect_calc_norm(__m256d x, __m256d y, __m256d z) {
        return _mm256_add_pd(_mm256_sqrt_pd(
                _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))),
                             d_epsilon_v);
    }

};

#endif //_AMPLITUDES_CALCULATOR_M256_H