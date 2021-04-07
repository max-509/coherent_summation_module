#ifndef KIRCHHOFF_MIGRATION_BY_POINTS_V_H_
#define KIRCHHOFF_MIGRATION_BY_POINTS_V_H_

#include "array2D.h"

#include <cstdint>
#include <type_traits>
#include <memory>
#include <vector>

template <typename T1, typename T2>
inline void process_receiver_data_on_grid(const T1 *curr_trace,
                                       const std::vector<T2> &times_to_source,
                                       const Array2D<T2> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       T1 *result_data,
                                       std::ptrdiff_t i_p_tail) {

    #pragma omp parallel for simd
    for (std::ptrdiff_t i_p = i_p_tail; i_p < n_points; ++i_p) {

        const T2 t_to_s = times_to_source[i_p], t_to_r = times_to_receivers(i_r, i_p);

        auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

        if (sample_idx < n_samples) {
            result_data[i_p] += curr_trace[sample_idx];
        }
    }
}

template <typename T1, typename T2>
inline void process_receiver_data_on_grid(const T1 *curr_trace,
                                       const std::vector<T2> &times_to_source,
                                       const Array2D<T2> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       T1 *result_data) {
    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, 0);
}

#ifdef __AVX512F__ //Selection of SIMD instructions
#include <immintrin.h>

//<float, float>
inline void process_receiver_data_on_grid(const float *curr_trace,
                                       const std::vector<float> &times_to_source,
                                       const Array2D<float> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       const std::ptrdiff_t &n_samples,
                                       float *result_data) {
    constexpr std::ptrdiff_t vector_dim = 16;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim);

    const __m512 rev_dt_v = _mm512_set1_ps(static_cast<float>(rev_dt));
    const __m512 n_samples_v = _mm512_set1_ps(static_cast<float>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        #ifdef __AVX512DQ__
        const __m256 zero_v = _mm256_setzero_ps();
        #else
        alignas(sizeof(__m512)) float i_samples[vector_dim];
        #endif //__AVX512DQ__

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim) {
            const __m512 t_to_s_v = _mm512_loadu_ps(times_to_source.data() + i_p),
                        t_to_r_v = _mm512_loadu_ps(&times_to_receivers(i_r, i_p));

            __m512 v_i_sample = _mm512_mul_ps(rev_dt_v, _mm512_add_ps(t_to_r_v, t_to_s_v));

            auto i_samples_mask = _mm512_cmp_ps_mask(n_samples_v, v_i_sample, _CMP_GT_OQ);

            #ifdef __AVX512DQ__
            auto gather_v_l = _mm512_mask_i64gather_ps(zero_v,
                                                         static_cast<__mmask8>(i_samples_mask & 0xFF),
                                                         _mm512_cvttps_epi64(_mm512_castps256_ps512(v_i_sample)),
                                                         curr_trace,
                                                         4);

            auto gather_v_h = _mm512_mask_i64gather_ps(zero_v,
                                                         static_cast<__mmask8>((i_samples_mask & 0xFF00) >> 8),
                                                         _mm512_cvttps_epi64(_mm512_extractf32x8_ps(v_i_sample, 1)),
                                                         curr_trace,
                                                         4);

            _mm512_storeu_ps(result_data + i_p, _mm512_add_ps(_mm512_loadu_ps(result_data + i_p),
                                                              _mm512_shuffle_f32x4(_mm512_castps256_ps512(gather_v_l),
                                                                                   _mm512_castps256_ps512(gather_v_h),
                                                                                   0x77)));
            #else
            _mm512_storeu_ps(i_samples, v_i_sample);

            if (i_samples_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_samples_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_samples_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_samples_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
            if (i_samples_mask & 0x10) {
                result_data[i_p + 4] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[4])];
            }
            if (i_samples_mask & 0x20) {
                result_data[i_p + 5] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[5])];
            }
            if (i_samples_mask & 0x40) {
                result_data[i_p + 6] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[6])];
            }
            if (i_samples_mask & 0x80) {
                result_data[i_p + 7] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[7])];
            }
            if (i_samples_mask & 0x100) {
                result_data[i_p + 8] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[8])];
            }
            if (i_samples_mask & 0x200) {
                result_data[i_p + 9] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[9])];
            }
            if (i_samples_mask & 0x400) {
                result_data[i_p + 10] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[10])];
            }
            if (i_samples_mask & 0x800) {
                result_data[i_p + 11] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[11])];
            }
            if (i_samples_mask & 0x1000) {
                result_data[i_p + 12] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[12])];
            }
            if (i_samples_mask & 0x2000) {
                result_data[i_p + 13] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[13])];
            }
            if (i_samples_mask & 0x4000) {
                result_data[i_p + 14] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[14])];
            }
            if (i_samples_mask & 0x8000) {
                result_data[i_p + 15] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[15])];
            }
            #endif //__AVX512DQ__
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<double, float>
inline void process_receiver_data_on_grid(const double *curr_trace,
                                       const std::vector<float> &times_to_source,
                                       const Array2D<float> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       double *result_data) {

    constexpr std::ptrdiff_t vector_dim_d = 8;
    constexpr std::ptrdiff_t vector_dim_f = 16;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim_f);

    const __m512 rev_dt_v = _mm512_set1_ps(static_cast<float>(rev_dt));
    const __m512 n_samples_v = _mm512_set1_ps(static_cast<float>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        #ifdef __AVX512DQ__
        const __m512d zero_v = _mm512_setzero_pd();
        #else
        alignas(sizeof(__m512)) float i_samples[vector_dim_f];
        #endif //__AVX512DQ__

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim_f) {
            const __m512 t_to_s_v = _mm512_loadu_ps(times_to_source.data() + i_p),
                        t_to_r_v = _mm512_loadu_ps(&times_to_receivers(i_r, i_p));

            __m512 v_i_sample = _mm512_mul_ps(rev_dt_v, _mm512_add_ps(t_to_r_v, t_to_s_v));

            auto i_samples_mask = _mm512_cmp_ps_mask(n_samples_v, v_i_sample, _CMP_GT_OQ);

            #ifdef __AVX512DQ__
            auto gather_v_l = _mm512_mask_i64gather_pd(zero_v,
                                                         static_cast<__mmask8>(i_samples_mask & 0xFF),
                                                         _mm512_cvttps_epi64(_mm512_castps256_ps512(v_i_sample)),
                                                         curr_trace,
                                                         8);

            _mm512_storeu_pd(result_data + i_p, _mm512_add_pd(_mm512_loadu_pd(result_data + i_p), gather_v_l));

            auto gather_v_h = _mm512_mask_i64gather_pd(zero_v,
                                                         static_cast<__mmask8>((i_samples_mask & 0xFF00) >> 8),
                                                         _mm512_cvttps_epi64(_mm512_extractf32x8_ps(v_i_sample, 1)),
                                                         curr_trace,
                                                         8);

            _mm512_storeu_pd(result_data + i_p + 4, _mm512_add_pd(_mm512_loadu_pd(result_data + i_p + 4), gather_v_h));
            #else
            _mm512_storeu_ps(i_samples, v_i_sample);

            if (i_samples_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_samples_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_samples_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_samples_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
            if (i_samples_mask & 0x10) {
                result_data[i_p + 4] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[4])];
            }
            if (i_samples_mask & 0x20) {
                result_data[i_p + 5] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[5])];
            }
            if (i_samples_mask & 0x40) {
                result_data[i_p + 6] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[6])];
            }
            if (i_samples_mask & 0x80) {
                result_data[i_p + 7] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[7])];
            }
            if (i_samples_mask & 0x100) {
                result_data[i_p + 8] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[8])];
            }
            if (i_samples_mask & 0x200) {
                result_data[i_p + 9] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[9])];
            }
            if (i_samples_mask & 0x400) {
                result_data[i_p + 10] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[10])];
            }
            if (i_samples_mask & 0x800) {
                result_data[i_p + 11] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[11])];
            }
            if (i_samples_mask & 0x1000) {
                result_data[i_p + 12] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[12])];
            }
            if (i_samples_mask & 0x2000) {
                result_data[i_p + 13] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[13])];
            }
            if (i_samples_mask & 0x4000) {
                result_data[i_p + 14] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[14])];
            }
            if (i_samples_mask & 0x8000) {
                result_data[i_p + 15] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[15])];
            }
            #endif //__AVX512DQ__
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<double, double>
inline void process_receiver_data_on_grid(const double *curr_trace,
                                       const std::vector<double> &times_to_source,
                                       const Array2D<double> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       double *result_data) {

    constexpr std::ptrdiff_t vector_dim = 8;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim);

    const __m512d rev_dt_v = _mm512_set1_pd(rev_dt);
    const __m512d n_samples_v = _mm512_set1_pd(static_cast<double>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        #ifdef __AVX512DQ__
        const __m512d zero_v = _mm512_setzero_pd();
        #else
        alignas(sizeof(__m512d)) double i_samples[vector_dim];
        #endif //__AVX512DQ__

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim) {
            const __m512d t_to_s_v = _mm512_loadu_pd(times_to_source.data() + i_p),
                        t_to_r_v = _mm512_loadu_pd(&times_to_receivers(i_r, i_p));

            __m512d v_i_sample = _mm512_mul_pd(rev_dt_v, _mm512_add_pd(t_to_r_v, t_to_s_v));
            auto i_samples_mask = _mm512_cmp_pd_mask(n_samples_v, v_i_sample, _CMP_GT_OQ);

            #ifdef __AVX512DQ__

            auto gather_v = _mm512_mask_i64gather_pd(zero_v, i_samples_mask, _mm512_cvttpd_epi64(v_i_sample), curr_trace, 8);
            _mm512_storeu_pd(result_data + i_p, _mm512_add_pd(_mm512_loadu_pd(result_data + i_p), gather_v));

            #else

            _mm512_storeu_pd(i_samples, v_i_sample);

            if (i_samples_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_samples_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_samples_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_samples_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
            if (i_samples_mask & 0x10) {
                result_data[i_p + 4] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[4])];
            }
            if (i_samples_mask & 0x20) {
                result_data[i_p + 5] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[5])];
            }
            if (i_samples_mask & 0x40) {
                result_data[i_p + 6] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[6])];
            }
            if (i_samples_mask & 0x80) {
                result_data[i_p + 7] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[7])];
            }

            #endif //__AVX512DQ__
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<float, double>
inline void process_receiver_data_on_grid(const float *curr_trace,
                                       const std::vector<double> &times_to_source,
                                       const Array2D<double> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       float *result_data) {

    constexpr std::ptrdiff_t vector_dim_d = 8;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim_d);

    const __m512d rev_dt_v = _mm512_set1_pd(rev_dt);
    const __m512d n_samples_v = _mm512_set1_pd(static_cast<double>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        #ifdef __AVX512DQ__
        const __m256 zero_v = _mm256_setzero_ps();
        #else
        alignas(sizeof(__m512d)) double i_samples[vector_dim_d];
        #endif //__AVX512DQ__

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim_d) {
            const __m512d t_to_s_v = _mm512_loadu_pd(times_to_source.data() + i_p),
                        t_to_r_v = _mm512_loadu_pd(&times_to_receivers(i_r, i_p));

            __m512d v_i_sample = _mm512_mul_pd(rev_dt_v, _mm512_add_pd(t_to_r_v, t_to_s_v));
            auto i_samples_mask = _mm512_cmp_pd_mask(n_samples_v, v_i_sample, _CMP_GT_OQ);

            #ifdef __AVX512DQ__

            auto gather_v = _mm512_mask_i64gather_ps(zero_v, i_samples_mask, _mm512_cvttpd_epi64(v_i_sample), curr_trace, 4);
            _mm256_storeu_ps(result_data + i_p, _mm256_add_ps(_mm256_loadu_ps(result_data + i_p), gather_v));

            #else

            _mm512_storeu_pd(i_samples, v_i_sample);

            if (i_samples_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_samples_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_samples_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_samples_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
            if (i_samples_mask & 0x10) {
                result_data[i_p + 4] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[4])];
            }
            if (i_samples_mask & 0x20) {
                result_data[i_p + 5] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[5])];
            }
            if (i_samples_mask & 0x40) {
                result_data[i_p + 6] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[6])];
            }
            if (i_samples_mask & 0x80) {
                result_data[i_p + 7] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[7])];
            }

            #endif //__AVX512DQ__
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

#elif __AVX2__
#include <immintrin.h>

//<float, float>
inline void process_receiver_data_on_grid(const float *curr_trace,
                                       const std::vector<float> &times_to_source,
                                       const Array2D<float> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       float *result_data) {

    constexpr std::ptrdiff_t vector_dim = 8;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim);

    const __m256 rev_dt_v = _mm256_set1_ps(static_cast<float>(rev_dt));
    const __m256 n_samples_v = _mm256_set1_ps(static_cast<float>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m256)) float i_samples[vector_dim];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim) {
            const __m256 t_to_s_v = _mm256_loadu_ps(times_to_source.data() + i_p),
                    t_to_r_v = _mm256_loadu_ps(&times_to_receivers(i_r, i_p));

            __m256 v_i_sample = _mm256_mul_ps(rev_dt_v, _mm256_add_ps(t_to_r_v, t_to_s_v));
            int i_sample_mask = _mm256_movemask_ps(_mm256_cmp_ps(n_samples_v, v_i_sample, _CMP_GT_OQ));
            _mm256_storeu_ps(i_samples, v_i_sample);

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_sample_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_sample_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
            if (i_sample_mask & 0x10) {
                result_data[i_p + 4] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[4])];
            }
            if (i_sample_mask & 0x20) {
                result_data[i_p + 5] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[5])];
            }
            if (i_sample_mask & 0x40) {
                result_data[i_p + 6] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[6])];
            }
            if (i_sample_mask & 0x80) {
                result_data[i_p + 7] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[7])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<double, float>
inline void process_receiver_data_on_grid(const double *curr_trace,
                                       const std::vector<float> &times_to_source,
                                       const Array2D<float> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       double *result_data) {

    constexpr std::ptrdiff_t vector_dim_d = 4;
    constexpr std::ptrdiff_t vector_dim_f = 8;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim_f);

    const __m256 rev_dt_v = _mm256_set1_ps(static_cast<float>(rev_dt));
    const __m256 n_samples_v = _mm256_set1_ps(static_cast<float>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m256)) float i_samples[vector_dim_f];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim_f) {
            const __m256 t_to_s_v = _mm256_loadu_ps(times_to_source.data() + i_p),
                    t_to_r_v = _mm256_loadu_ps(&times_to_receivers(i_r, i_p));

            __m256 v_i_sample = _mm256_mul_ps(rev_dt_v, _mm256_add_ps(t_to_r_v, t_to_s_v));
            int i_sample_mask = _mm256_movemask_ps(_mm256_cmp_ps(n_samples_v, v_i_sample, _CMP_GT_OQ));
            _mm256_storeu_ps(i_samples, v_i_sample);

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_sample_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_sample_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
            if (i_sample_mask & 0x10) {
                result_data[i_p + 4] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[4])];
            }
            if (i_sample_mask & 0x20) {
                result_data[i_p + 5] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[5])];
            }
            if (i_sample_mask & 0x40) {
                result_data[i_p + 6] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[6])];
            }
            if (i_sample_mask & 0x80) {
                result_data[i_p + 7] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[7])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<double, double>
inline void process_receiver_data_on_grid(const double *curr_trace,
                                       const std::vector<double> &times_to_source,
                                       const Array2D<double> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       double *result_data) {

    constexpr std::ptrdiff_t vector_dim = 4;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim);

    const __m256d rev_dt_v = _mm256_set1_pd(rev_dt);
    const __m256d n_samples_v = _mm256_set1_pd(static_cast<double>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m256d)) double i_samples[vector_dim];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim) {
            const __m256d t_to_s_v = _mm256_loadu_pd(times_to_source.data() + i_p),
                    t_to_r_v = _mm256_loadu_pd(&times_to_receivers(i_r, i_p));

            __m256d v_i_sample = _mm256_mul_pd(rev_dt_v, _mm256_add_pd(t_to_r_v, t_to_s_v));
            int i_sample_mask = _mm256_movemask_pd(_mm256_cmp_pd(n_samples_v, v_i_sample, _CMP_GT_OQ));
            _mm256_storeu_pd(i_samples, v_i_sample);

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_sample_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_sample_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<float, double>
inline void process_receiver_data_on_grid(const float *curr_trace,
                                       const std::vector<double> &times_to_source,
                                       const Array2D<double> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       float *result_data) {

    constexpr std::ptrdiff_t vector_dim_d = 4;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim_d);

    const __m256d rev_dt_v = _mm256_set1_pd(rev_dt);
    const __m256d n_samples_v = _mm256_set1_pd(static_cast<double>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m256d)) double i_samples[vector_dim_d];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim_d) {
            const __m256d t_to_s_v = _mm256_loadu_pd(times_to_source.data() + i_p),
                    t_to_r_v = _mm256_loadu_pd(&times_to_receivers(i_r, i_p));

            __m256d v_i_sample = _mm256_mul_pd(rev_dt_v, _mm256_add_pd(t_to_r_v, t_to_s_v));
            int i_sample_mask = _mm256_movemask_pd(_mm256_cmp_pd(n_samples_v, v_i_sample, _CMP_GT_OQ));
            _mm256_storeu_pd(i_samples, v_i_sample);

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_sample_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_sample_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

#elif __SSE4_2__
#include <immintrin.h>

//<float, float>
inline void process_receiver_data_on_grid(const float *curr_trace,
                                       const std::vector<float> &times_to_source,
                                       const Array2D<float> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       float *result_data) {

    constexpr std::ptrdiff_t vector_dim = 4;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim);

    const __m128 rev_dt_v = _mm_set1_ps(static_cast<float>(rev_dt));
    const __m128 n_samples_v = _mm_set1_ps(static_cast<float>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m128)) float i_samples[vector_dim];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim) {
            const __m128 t_to_s_v = _mm_loadu_ps(times_to_source.data() + i_p),
                        t_to_r_v = _mm_loadu_ps(&times_to_receivers(i_r, i_p));

            __m128 v_i_sample_f = _mm_mul_ps(rev_dt_v, _mm_add_ps(t_to_r_v, t_to_s_v));

            _mm_storeu_ps(i_samples, v_i_sample_f);
            int i_sample_mask = _mm_movemask_ps(_mm_cmpgt_ps(n_samples_v, v_i_sample_f));

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_sample_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_sample_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<double, float>
inline void process_receiver_data_on_grid(const double *curr_trace,
                                       const std::vector<float> &times_to_source,
                                       const Array2D<float> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       double *result_data) {

    constexpr std::ptrdiff_t vector_dim_f = 4;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim_f);

    const __m128 rev_dt_v = _mm_set1_ps(static_cast<float>(rev_dt));
    const __m128 n_samples_v = _mm_set1_ps(static_cast<float>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m128)) float i_samples[vector_dim_f];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim_f) {
            const __m128 t_to_s_v = _mm_loadu_ps(times_to_source.data() + i_p),
                        t_to_r_v = _mm_loadu_ps(&times_to_receivers(i_r, i_p));

            __m128 v_i_sample_f = _mm_mul_ps(rev_dt_v, _mm_add_ps(t_to_r_v, t_to_s_v));

            _mm_storeu_ps(i_samples, v_i_sample_f);
            int i_sample_mask = _mm_movemask_ps(_mm_cmpgt_ps(n_samples_v, v_i_sample_f));

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
            if (i_sample_mask & 0x4) {
                result_data[i_p + 2] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[2])];
            }
            if (i_sample_mask & 0x8) {
                result_data[i_p + 3] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[3])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<double, double>
inline void process_receiver_data_on_grid(const double *curr_trace,
                                       const std::vector<double> &times_to_source,
                                       const Array2D<double> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       double *result_data) {

    constexpr std::ptrdiff_t vector_dim = 2;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim);

    const __m128d rev_dt_v = _mm_set1_pd(rev_dt);
    const __m128d n_samples_v = _mm_set1_pd(static_cast<double>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {

        alignas(sizeof(__m128d)) double i_samples[vector_dim];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim) {
            const __m128d t_to_s_v = _mm_loadu_pd(times_to_source.data() + i_p),
                        t_to_r_v = _mm_loadu_pd(&times_to_receivers(i_r, i_p));

            __m128d v_i_sample_f = _mm_mul_pd(rev_dt_v, _mm_add_pd(t_to_r_v, t_to_s_v));

            _mm_storeu_pd(i_samples, v_i_sample_f);
            int i_sample_mask = _mm_movemask_pd(_mm_cmpgt_pd(n_samples_v, v_i_sample_f));

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

//<float, double>
inline void process_receiver_data_on_grid(const float *curr_trace,
                                       const std::vector<double> &times_to_source,
                                       const Array2D<double> &times_to_receivers,
                                       const std::ptrdiff_t &n_points,
                                       const std::ptrdiff_t &n_samples,
                                       const double &rev_dt,
                                       const std::ptrdiff_t &i_r,
                                       float *result_data) {
    constexpr std::ptrdiff_t vector_dim_d = 2;
    const std::ptrdiff_t n_points_without_remainder = n_points - (n_points % vector_dim_d);

    const __m128d rev_dt_v = _mm_set1_pd(rev_dt);
    const __m128d n_samples_v = _mm_set1_pd(static_cast<double>(n_samples));
    #pragma omp parallel shared(rev_dt_v, n_samples_v, n_points_without_remainder)
    {
        alignas(sizeof(__m128d)) double i_samples[vector_dim_d];

        #pragma omp for
        for (auto i_p = 0; i_p < n_points_without_remainder; i_p += vector_dim_d) {
            const __m128d t_to_s_v = _mm_loadu_pd(times_to_source.data() + i_p),
                        t_to_r_v = _mm_loadu_pd(&times_to_receivers(i_r, i_p));

            __m128d v_i_sample_f = _mm_mul_pd(rev_dt_v, _mm_add_pd(t_to_r_v, t_to_s_v));

            _mm_storeu_pd(i_samples, v_i_sample_f);
            int i_sample_mask = _mm_movemask_pd(_mm_cmpgt_pd(n_samples_v, v_i_sample_f));

            if (i_sample_mask & 0x1) {
                result_data[i_p] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[0])];
            }
            if (i_sample_mask & 0x2) {
                result_data[i_p + 1] += curr_trace[static_cast<std::ptrdiff_t>(i_samples[1])];
            }
        }
    }

    process_receiver_data_on_grid(curr_trace, times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data, n_points_without_remainder);
}

#endif //End selection of SIMD instructions

#endif //KIRCHHOFF_MIGRATION_BY_POINTS_V_H_
