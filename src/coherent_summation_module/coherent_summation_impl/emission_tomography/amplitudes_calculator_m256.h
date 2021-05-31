#ifndef _AMPLITUDES_CALCULATOR_M256_H
#define _AMPLITUDES_CALCULATOR_M256_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"
#include "array1D.h"

#include <immintrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

template<typename T,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
class AmplitudesCalculatorM256 : public AmplitudesCalculatorBase<T, AmplitudesCalculatorM256<T>> {
public:
    using value_type = T;
    using size_type = std::ptrdiff_t;

    AmplitudesCalculatorM256(const Array2D<value_type> &sources_coords,
                             const Array1D<value_type> &tensor_matrix) :
            AmplitudesCalculatorBase<T, AmplitudesCalculatorM256>(sources_coords, tensor_matrix) {}

    friend AmplitudesCalculatorBase<T, AmplitudesCalculatorM256<T>>;

private:
    __m256d abs_mask_d = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    __m256 abs_mask_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256d d_epsilon_v = _mm256_set1_pd(std::numeric_limits<double>::epsilon());
    __m256 f_epsilon_v = _mm256_set1_ps(std::numeric_limits<float>::epsilon());

    void realize_calculate(const Array2D<value_type> &rec_coords_, Array2D<value_type> &amplitudes_) {
        realize_calculate_impl(rec_coords_, amplitudes_);
    }

    void realize_calculate_impl(const Array2D<float> &rec_coords_, Array2D<float> &amplitudes_) {
        static_assert(std::is_same<float, value_type>::value,
                      "Error: In amplitudes calculator SIMD implementation types input and output arrays must be equal");
        auto n_rec = rec_coords_.get_y_dim();
        auto sources_count = this->sources_coords_.get_y_dim();
        auto rec_coords_y_stride = rec_coords_.get_y_stride();
        constexpr size_type matrix_size = 6;
        constexpr size_type vector_dim = sizeof(__m256) / sizeof(float);

        static const __m256 two_v = _mm256_set1_ps(2.0f);
        __m256 tensor_matrix_v[matrix_size] = {_mm256_broadcast_ss(this->tensor_matrix_.get(0)),
                                               _mm256_broadcast_ss(this->tensor_matrix_.get(1)),
                                               _mm256_broadcast_ss(this->tensor_matrix_.get(2)),
                                               _mm256_broadcast_ss(this->tensor_matrix_.get(3)),
                                               _mm256_broadcast_ss(this->tensor_matrix_.get(4)),
                                               _mm256_broadcast_ss(this->tensor_matrix_.get(5))
        };
        __m256i vindex = _mm256_set_epi32(rec_coords_y_stride * 7,
                                          rec_coords_y_stride * 6,
                                          rec_coords_y_stride * 5,
                                          rec_coords_y_stride * 4,
                                          rec_coords_y_stride * 3,
                                          rec_coords_y_stride * 2,
                                          rec_coords_y_stride * 1,
                                          rec_coords_y_stride * 0);

#pragma omp parallel
        {
            __m256 coord_vec[3];

            #pragma omp for schedule(static) collapse(2)
            for (size_type i = 0; i < sources_count; ++i) {
                for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                    for (size_type crd = 0; crd < 3; ++crd) {
                        coord_vec[crd] = _mm256_sub_ps(
                                _mm256_i32gather_ps(rec_coords_.get(r_ind, crd), vindex, sizeof(float)),
                                _mm256_broadcast_ss(this->sources_coords_.get(i, crd)));
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

                    _mm256_storeu_ps(amplitudes_.get(i, r_ind), _mm256_div_ps(ampl_vect,
                                                                              _mm256_add_ps(
                                                                                      _mm256_and_ps(ampl_vect, abs_mask_f),
                                                                                      f_epsilon_v)));
                }
            }
        }

        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), this->sources_coords_, rec_coords_,
                                              this->tensor_matrix_,
                                              amplitudes_);
    }

    void realize_calculate_impl(const Array2D<double> &rec_coords_, Array2D<double> &amplitudes_) {
        static_assert(std::is_same<double, value_type>::value,
                      "Error: In amplitudes calculator SIMD implementation types input and output arrays must be equal");
        auto n_rec = rec_coords_.get_y_dim();
        auto sources_count = this->sources_coords_.get_y_dim();
        auto rec_coords_y_stride = rec_coords_.get_y_stride();
        constexpr size_type matrix_size = 6;
        constexpr size_type vector_dim = sizeof(__m256d) / sizeof(double);

        static const __m256d two_v = _mm256_set1_pd(2.0);
        static const __m256d one_v = _mm256_set1_pd(1.0);
        __m256d tensor_matrix_v[matrix_size] = {_mm256_broadcast_sd(this->tensor_matrix_.get(0)),
                                                _mm256_broadcast_sd(this->tensor_matrix_.get(1)),
                                                _mm256_broadcast_sd(this->tensor_matrix_.get(2)),
                                                _mm256_broadcast_sd(this->tensor_matrix_.get(3)),
                                                _mm256_broadcast_sd(this->tensor_matrix_.get(4)),
                                                _mm256_broadcast_sd(this->tensor_matrix_.get(5))
        };

        __m256i vindex = _mm256_set_epi64x(rec_coords_y_stride * 3,
                                           rec_coords_y_stride * 2,
                                           rec_coords_y_stride * 1,
                                           rec_coords_y_stride * 0);


#pragma omp parallel
        {
            __m256d coord_vec[3];

            #pragma omp for schedule(static) collapse(2)
            for (size_type i = 0; i < sources_count; ++i) {
                for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                    for (size_type crd = 0; crd < 3; ++crd) {
                        coord_vec[crd] = _mm256_sub_pd(
                                _mm256_i64gather_pd(rec_coords_.get(r_ind, crd), vindex, sizeof(double)),
                                _mm256_broadcast_sd(this->sources_coords_.get(i, crd)));
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

                    _mm256_storeu_pd(amplitudes_.get(i, r_ind), _mm256_div_pd(ampl_vect,
                                                                              _mm256_add_pd(
                                                                                      _mm256_and_pd(ampl_vect, abs_mask_d),
                                                                                      d_epsilon_v)));
                }
            }
        }

        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), this->sources_coords_, rec_coords_,
                                              this->tensor_matrix_,
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
