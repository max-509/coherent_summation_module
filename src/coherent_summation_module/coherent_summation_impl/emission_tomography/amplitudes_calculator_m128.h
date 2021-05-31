#ifndef _AMPLITUDES_CALCULATOR_M128_H
#define _AMPLITUDES_CALCULATOR_M128_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <immintrin.h>
#include <functional>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <type_traits>

template<typename T,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
class AmplitudesCalculatorM128 : public AmplitudesCalculatorBase<T, AmplitudesCalculatorM128<T>> {
public:
    using value_type = T;
    using size_type = std::ptrdiff_t;

    AmplitudesCalculatorM128(const Array2D<value_type> &sources_coords,
                             const Array1D<value_type> &tensor_matrix) :
            AmplitudesCalculatorBase<T, AmplitudesCalculatorM128>(sources_coords, tensor_matrix) {}

    friend AmplitudesCalculatorBase<T, AmplitudesCalculatorM128<T>>;

private:
    __m128d abs_mask_d = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    __m128 abs_mask_f = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128d d_epsilon_v = _mm_set1_pd(std::numeric_limits<double>::epsilon());
    __m128 f_epsilon_v = _mm_set1_ps(std::numeric_limits<float>::epsilon());

    void realize_calculate(const Array2D<value_type> &rec_coords_, Array2D<value_type> &amplitudes_) {
        realize_calculate_impl(rec_coords_, amplitudes_);
    }

    void realize_calculate_impl(const Array2D<float> &rec_coords_, Array2D<float> &amplitudes_) {
        static_assert(std::is_same<float, value_type>::value,
                      "Error: In amplitudes calculator SIMD implementation types input and output arrays must be equal");
        auto n_rec = rec_coords_.get_y_dim();
        auto sources_count = this->sources_coords_.get_y_dim();
        constexpr size_type matrix_size = 6;
        constexpr size_type vector_dim = sizeof(__m128) / sizeof(float);

        static __m128 two_v = _mm_set1_ps(2.0f);
        __m128 tensor_matrix_v[matrix_size] = {_mm_set1_ps(this->tensor_matrix_[0]),
                                               _mm_set1_ps(this->tensor_matrix_[1]),
                                               _mm_set1_ps(this->tensor_matrix_[2]),
                                               _mm_set1_ps(this->tensor_matrix_[3]),
                                               _mm_set1_ps(this->tensor_matrix_[4]),
                                               _mm_set1_ps(this->tensor_matrix_[5])
        };

#pragma omp parallel
        {

            __m128 coord_vec[3];

            #pragma omp for schedule(static) collapse(2)
            for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
                for (std::ptrdiff_t r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                    for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                        coord_vec[crd] = _mm_sub_ps(_mm_set_ps(rec_coords_(r_ind + 3, crd), rec_coords_(r_ind + 2, crd),
                                                               rec_coords_(r_ind + 1, crd), rec_coords_(r_ind + 0, crd)),
                                                    _mm_set1_ps(this->sources_coords_(i, crd)));
                    }

                    __m128 rev_dist = _mm_rcp_ps(vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                    coord_vec[2] = _mm_mul_ps(coord_vec[2], rev_dist);

                    __m128 norm_coord_z = _mm_mul_ps(coord_vec[2], rev_dist);

                    __m128 ampl_vect = _mm_mul_ps(tensor_matrix_v[2],
                                                  _mm_mul_ps(norm_coord_z, _mm_mul_ps(coord_vec[2], coord_vec[2])));

                    coord_vec[1] = _mm_mul_ps(coord_vec[1], rev_dist);

    #ifdef __FMA__
                    ampl_vect = _mm_fmadd_ps(tensor_matrix_v[1], _mm_mul_ps(norm_coord_z, _mm_mul_ps(coord_vec[1], coord_vec[1])), ampl_vect);
                    coord_vec[0] = _mm_mul_ps(coord_vec[0], rev_dist);
                    ampl_vect = _mm_fmadd_ps(tensor_matrix_v[0], _mm_mul_ps(norm_coord_z, _mm_mul_ps(coord_vec[0], coord_vec[0])), ampl_vect);

                    __m128 double_norm_coord_z = _mm_mul_ps(norm_coord_z, two_v);

                    ampl_vect = _mm_fmadd_ps(tensor_matrix_v[3], _mm_mul_ps(double_norm_coord_z, _mm_mul_ps(coord_vec[1], coord_vec[2])), ampl_vect);
                    ampl_vect = _mm_fmadd_ps(tensor_matrix_v[4], _mm_mul_ps(double_norm_coord_z, _mm_mul_ps(coord_vec[0], coord_vec[2])), ampl_vect);
                    ampl_vect = _mm_fmadd_ps(tensor_matrix_v[5], _mm_mul_ps(double_norm_coord_z, _mm_mul_ps(coord_vec[0], coord_vec[1])), ampl_vect);
    #else
                    ampl_vect = _mm_add_ps(ampl_vect, _mm_mul_ps(tensor_matrix_v[1], _mm_mul_ps(norm_coord_z,
                                                                                                _mm_mul_ps(coord_vec[1],
                                                                                                           coord_vec[1]))));
                    coord_vec[0] = _mm_mul_ps(coord_vec[0], rev_dist);
                    ampl_vect = _mm_add_ps(ampl_vect, _mm_mul_ps(tensor_matrix_v[0], _mm_mul_ps(norm_coord_z,
                                                                                                _mm_mul_ps(coord_vec[0],
                                                                                                           coord_vec[0]))));

                    __m128 double_norm_coord_z = _mm_mul_ps(norm_coord_z, two_v);

                    ampl_vect = _mm_add_ps(ampl_vect, _mm_mul_ps(tensor_matrix_v[3], _mm_mul_ps(double_norm_coord_z,
                                                                                                _mm_mul_ps(coord_vec[1],
                                                                                                           coord_vec[2]))));
                    ampl_vect = _mm_add_ps(ampl_vect, _mm_mul_ps(tensor_matrix_v[4], _mm_mul_ps(double_norm_coord_z,
                                                                                                _mm_mul_ps(coord_vec[0],
                                                                                                           coord_vec[2]))));
                    ampl_vect = _mm_add_ps(ampl_vect, _mm_mul_ps(tensor_matrix_v[5], _mm_mul_ps(double_norm_coord_z,
                                                                                                _mm_mul_ps(coord_vec[0],
                                                                                                           coord_vec[1]))));
    #endif

                    _mm_storeu_ps(amplitudes_.get(i, r_ind),
                                  _mm_div_ps(ampl_vect, _mm_add_ps(_mm_and_ps(ampl_vect, abs_mask_f), f_epsilon_v)));
                }
            } 
        }
        
        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), this->sources_coords_, rec_coords_,
                                              this->tensor_matrix_, amplitudes_);
    }

    void realize_calculate_impl(const Array2D<double> &rec_coords_, Array2D<double> &amplitudes_) {
        static_assert(std::is_same<double, value_type>::value,
                      "Error: In amplitudes calculator SIMD implementation types input and output arrays must be equal");
        auto n_rec = rec_coords_.get_y_dim();
        auto sources_count = this->sources_coords_.get_y_dim();
        constexpr size_type matrix_size = 6;
        constexpr size_type vector_dim = sizeof(__m128d) / sizeof(double);

        static const __m128d two_v = _mm_set1_pd(2.0);
        static const __m128d one_v = _mm_set1_pd(1.0);
        __m128d tensor_matrix_v[matrix_size] = {_mm_set1_pd(this->tensor_matrix_[0]),
                                                _mm_set1_pd(this->tensor_matrix_[1]),
                                                _mm_set1_pd(this->tensor_matrix_[2]),
                                                _mm_set1_pd(this->tensor_matrix_[3]),
                                                _mm_set1_pd(this->tensor_matrix_[4]),
                                                _mm_set1_pd(this->tensor_matrix_[5])
        };

#pragma omp parallel
        {
            __m128d coord_vec[3];

            #pragma omp for schedule(static) collapse(2)
            for (std::ptrdiff_t i = 0; i < sources_count; ++i) {
                for (std::ptrdiff_t r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                    for (std::ptrdiff_t crd = 0; crd < 3; ++crd) {
                        coord_vec[crd] = _mm_sub_pd(_mm_set_pd(rec_coords_(r_ind + 1, crd), rec_coords_(r_ind + 0, crd)),
                                                    _mm_set1_pd(this->sources_coords_(i, crd)));
                    }

                    __m128d rev_dist = _mm_div_pd(one_v, vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]));

                    coord_vec[2] = _mm_mul_pd(coord_vec[2], rev_dist);

                    __m128d norm_coord_z = _mm_mul_pd(coord_vec[2], rev_dist);

                    __m128d ampl_vect = _mm_mul_pd(tensor_matrix_v[2],
                                                   _mm_mul_pd(norm_coord_z, _mm_mul_pd(coord_vec[2], coord_vec[2])));

                    coord_vec[1] = _mm_mul_pd(coord_vec[1], rev_dist);

    #ifdef __FMA__
                    ampl_vect = _mm_fmadd_pd(tensor_matrix_v[1], _mm_mul_pd(norm_coord_z, _mm_mul_pd(coord_vec[1], coord_vec[1])), ampl_vect);
                    coord_vec[0] = _mm_mul_pd(coord_vec[0], rev_dist);
                    ampl_vect = _mm_fmadd_pd(tensor_matrix_v[0], _mm_mul_pd(norm_coord_z, _mm_mul_pd(coord_vec[0], coord_vec[0])), ampl_vect);

                    __m128d double_norm_coord_z = _mm_mul_pd(norm_coord_z, two_v);

                    ampl_vect = _mm_fmadd_pd(tensor_matrix_v[3], _mm_mul_pd(double_norm_coord_z, _mm_mul_pd(coord_vec[1], coord_vec[2])), ampl_vect);
                    ampl_vect = _mm_fmadd_pd(tensor_matrix_v[4], _mm_mul_pd(double_norm_coord_z, _mm_mul_pd(coord_vec[0], coord_vec[2])), ampl_vect);
                    ampl_vect = _mm_fmadd_pd(tensor_matrix_v[5], _mm_mul_pd(double_norm_coord_z, _mm_mul_pd(coord_vec[0], coord_vec[1])), ampl_vect);
    #else
                    ampl_vect = _mm_add_pd(ampl_vect, _mm_mul_pd(tensor_matrix_v[1], _mm_mul_pd(norm_coord_z,
                                                                                                _mm_mul_pd(coord_vec[1],
                                                                                                           coord_vec[1]))));
                    coord_vec[0] = _mm_mul_pd(coord_vec[0], rev_dist);
                    ampl_vect = _mm_add_pd(ampl_vect, _mm_mul_pd(tensor_matrix_v[0], _mm_mul_pd(norm_coord_z,
                                                                                                _mm_mul_pd(coord_vec[0],
                                                                                                           coord_vec[0]))));

                    __m128d double_norm_coord_z = _mm_mul_pd(norm_coord_z, two_v);

                    ampl_vect = _mm_add_pd(ampl_vect, _mm_mul_pd(tensor_matrix_v[3], _mm_mul_pd(double_norm_coord_z,
                                                                                                _mm_mul_pd(coord_vec[1],
                                                                                                           coord_vec[2]))));
                    ampl_vect = _mm_add_pd(ampl_vect, _mm_mul_pd(tensor_matrix_v[4], _mm_mul_pd(double_norm_coord_z,
                                                                                                _mm_mul_pd(coord_vec[0],
                                                                                                           coord_vec[2]))));
                    ampl_vect = _mm_add_pd(ampl_vect, _mm_mul_pd(tensor_matrix_v[5], _mm_mul_pd(double_norm_coord_z,
                                                                                                _mm_mul_pd(coord_vec[0],
                                                                                                           coord_vec[1]))));
    #endif

                    _mm_storeu_pd(amplitudes_.get(i, r_ind),
                                  _mm_div_pd(ampl_vect, _mm_add_pd(_mm_and_pd(ampl_vect, abs_mask_d), d_epsilon_v)));
                }
            }
        }
        

        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), this->sources_coords_, rec_coords_,
                                              this->tensor_matrix_, amplitudes_);
    }

    inline __m128 vect_calc_norm(__m128 x, __m128 y, __m128 z) {
        return _mm_add_ps(_mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z))),
                          f_epsilon_v);
    }

    inline __m128d vect_calc_norm(__m128d x, __m128d y, __m128d z) {
        return _mm_add_pd(_mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), _mm_mul_pd(z, z))),
                          d_epsilon_v);
    }

};

#endif //_AMPLITUDES_CALCULATOR_M128_H