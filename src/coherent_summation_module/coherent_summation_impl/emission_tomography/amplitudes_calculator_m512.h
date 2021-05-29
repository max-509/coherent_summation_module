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

template<typename T,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
class AmplitudesCalculatorM512 : public AmplitudesCalculatorBase<T, AmplitudesCalculatorM512<T>> {
public:

    using value_type = T;
    using size_type = std::ptrdiff_t;

    AmplitudesCalculatorM512(const Array2D<value_type> &sources_coords,
                             const Array1D<value_type> &tensor_matrix) :
            AmplitudesCalculatorBase<T, AmplitudesCalculatorM512>(sources_coords, tensor_matrix) {}

    friend AmplitudesCalculatorBase<T, AmplitudesCalculatorM512<T>>;

private:
    __m512d d_epsilon_v = _mm512_set1_pd(std::numeric_limits<double>::epsilon());
    __m512 f_epsilon_v = _mm512_set1_ps(std::numeric_limits<float>::epsilon());

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
        constexpr size_type vector_dim = sizeof(__m512) / sizeof(float);

        static const __m512 two_v = _mm512_set1_ps(2.0f);
        __m512 tensor_matrix_v[matrix_size] = {_mm512_set1_ps(this->tensor_matrix_[0]),
                                               _mm512_set1_ps(this->tensor_matrix_[1]),
                                               _mm512_set1_ps(this->tensor_matrix_[2]),
                                               _mm512_set1_ps(this->tensor_matrix_[3]),
                                               _mm512_set1_ps(this->tensor_matrix_[4]),
                                               _mm512_set1_ps(this->tensor_matrix_[5])
        };
        __m512i vindex = _mm512_set_epi32(rec_coords_y_stride * 15,
                                          rec_coords_y_stride * 14,
                                          rec_coords_y_stride * 13,
                                          rec_coords_y_stride * 12,
                                          rec_coords_y_stride * 11,
                                          rec_coords_y_stride * 10,
                                          rec_coords_y_stride * 9,
                                          rec_coords_y_stride * 8,
                                          rec_coords_y_stride * 7,
                                          rec_coords_y_stride * 6,
                                          rec_coords_y_stride * 5,
                                          rec_coords_y_stride * 4,
                                          rec_coords_y_stride * 3,
                                          rec_coords_y_stride * 2,
                                          rec_coords_y_stride * 1,
                                          rec_coords_y_stride * 0);

        __m512 coord_vec[3];

#ifdef _MSC_VER
#pragma omp parallel for schedule(static) collapse(2)
#else
#pragma omp parallel for simd schedule(static) collapse(2)
#endif
        for (size_type i = 0; i < sources_count; ++i) {
            for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                for (size_type crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_ps(
                            _mm512_i32gather_ps(vindex, rec_coords_.get(r_ind, crd), sizeof(float)),
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

                _mm512_storeu_ps(amplitudes_.get(i, r_ind),
                                 _mm512_div_ps(ampl_vect, _mm512_add_ps(_mm512_abs_ps(ampl_vect), f_epsilon_v)));
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
        constexpr size_type vector_dim = sizeof(__m512d) / sizeof(double);

        static const __m512d two_v = _mm512_set1_pd(2.0);
        __m512d tensor_matrix_v[matrix_size] = {_mm512_set1_pd(this->tensor_matrix_[0]),
                                                _mm512_set1_pd(this->tensor_matrix_[1]),
                                                _mm512_set1_pd(this->tensor_matrix_[2]),
                                                _mm512_set1_pd(this->tensor_matrix_[3]),
                                                _mm512_set1_pd(this->tensor_matrix_[4]),
                                                _mm512_set1_pd(this->tensor_matrix_[5])
        };
        __m512i vindex = _mm512_set_epi64(rec_coords_y_stride * 7,
                                          rec_coords_y_stride * 6,
                                          rec_coords_y_stride * 5,
                                          rec_coords_y_stride * 4,
                                          rec_coords_y_stride * 3,
                                          rec_coords_y_stride * 2,
                                          rec_coords_y_stride * 1,
                                          rec_coords_y_stride * 0);

        __m512d coord_vec[3];

#ifdef _MSC_VER
#pragma omp parallel for schedule(static) collapse(2)
#else
#pragma omp parallel for simd schedule(static) collapse(2)
#endif
        for (size_type i = 0; i < sources_count; ++i) {
            for (size_type r_ind = 0; r_ind < n_rec - (n_rec % vector_dim); r_ind += vector_dim) {
                for (size_type crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_pd(
                            _mm512_i64gather_pd(vindex, rec_coords_.get(r_ind, crd), sizeof(double)),
                            _mm512_set1_pd(this->sources_coords_(i, crd)));
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

                _mm512_storeu_pd(amplitudes_.get(i, r_ind),
                                 _mm512_div_pd(ampl_vect, _mm512_add_pd(_mm512_abs_pd(ampl_vect), d_epsilon_v)));
            }
        }
        this->non_vector_calculate_amplitudes(n_rec - (n_rec % vector_dim), this->sources_coords_, rec_coords_,
                                              this->tensor_matrix_,
                                              amplitudes_);
    }

    inline __m512 vect_calc_norm(__m512 x, __m512 y, __m512 z) {
        return _mm512_add_ps(_mm512_sqrt_ps(
                _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, x), _mm512_mul_ps(y, y)), _mm512_mul_ps(z, z))),
                             f_epsilon_v);
    }

    inline __m512d vect_calc_norm(__m512d x, __m512d y, __m512d z) {
        return _mm512_add_pd(_mm512_sqrt_pd(
                _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)), _mm512_mul_pd(z, z))),
                             d_epsilon_v);
    }

};

#endif //_AMPLITUDES_CALCULATOR_M512_H