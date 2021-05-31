#ifndef MINMAX_T_TO_SOURCE_H
#define MINMAX_T_TO_SOURCE_H

#include "array2D.h"

#include <cstddef>
#include <tuple>
#include <algorithm>
#include <cmath>

template<typename T>
inline std::pair<T, T>
find_minmax_t_to_source(const Array2D<T> &sources_receivers_times,
                        std::ptrdiff_t i_s,
                        std::ptrdiff_t i_begin,
                        std::ptrdiff_t i_end) {
    auto min = sources_receivers_times(i_s, i_begin);
    auto max = min;
    for (auto i_r = i_begin + 1; i_r < i_end; ++i_r) {
        auto curr_el = sources_receivers_times(i_s, i_r);
        if (curr_el > max) {
            max = curr_el;
        }
        if (curr_el < min) {
            min = curr_el;
        }
    }

    return std::make_pair(min, max);
}

template<typename T>
inline std::pair<T, T>
find_minmax_t_to_source_sequential(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s);

template<typename T>
inline std::pair<T, T>
find_minmax_t_to_source_stride(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s);

//Selection of SIMD instructions
#ifdef __AVX512F__ // AVX512 SIMD

#include <immintrin.h>

template<>
inline std::pair<double, double>
find_minmax_t_to_source_sequential(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 8;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        auto minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, 0), sources_receivers_times.get(i_s, n_receivers));
        return std::make_pair(*(minmax_iterators.first), *(minmax_iterators.second));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto min_elements_v = _mm512_loadu_pd(sources_receivers_times.get(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm512_loadu_pd(sources_receivers_times.get(i_s, i_r + vector_dim));
        min_elements_v = _mm512_min_pd(min_elements_v, new_elements);
        max_elements_v = _mm512_max_pd(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, n_receivers_multiple_vector_dim), sources_receivers_times.get(i_s, n_receivers));

    alignas(sizeof(__m512d)) double v_buffer[vector_dim];

    _mm512_storeu_pd(v_buffer, min_elements_v);
    double min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.first)
    );

    _mm512_storeu_pd(v_buffer, max_elements_v);
    double max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.second)
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<float, float>
find_minmax_t_to_source_sequential(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 16;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        auto minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, 0), sources_receivers_times.get(i_s, n_receivers));
        return std::make_pair(*(minmax_iterators.first), *(minmax_iterators.second));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto min_elements_v = _mm512_loadu_ps(sources_receivers_times.get(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm512_loadu_ps(sources_receivers_times.get(i_s, i_r + vector_dim));
        min_elements_v = _mm512_min_ps(min_elements_v, new_elements);
        max_elements_v = _mm512_max_ps(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, n_receivers_multiple_vector_dim), sources_receivers_times.get(i_s, n_receivers));

    alignas(sizeof(__m512)) float v_buffer[vector_dim];

    _mm512_storeu_ps(v_buffer, min_elements_v);
    float min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.first)
    );

    _mm512_storeu_ps(v_buffer, max_elements_v);
    float max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.second)
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<double, double>
find_minmax_t_to_source_stride(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 8;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return find_minmax_t_to_source(sources_receivers_times,
                                       i_s,
                                       0,
                                       n_receivers);
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto x_stride = sources_receivers_times.get_x_stride();

    const __m512i vindex = _mm512_set_epi64(x_stride * 7,
                                            x_stride * 6,
                                            x_stride * 5,
                                            x_stride * 4,
                                            x_stride * 3,
                                            x_stride * 2,
                                            x_stride * 1,
                                            x_stride * 0);

    auto min_elements_v = _mm512_i64gather_pd(vindex,
                                              sources_receivers_times.get(i_s, 0),
                                              sizeof(double));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm512_i64gather_pd(vindex,
                                                sources_receivers_times.get(i_s, i_r + vector_dim),
                                                sizeof(double));
        min_elements_v = _mm512_min_pd(min_elements_v, new_elements);
        max_elements_v = _mm512_max_pd(max_elements_v, new_elements);
    }

    auto remainder_minmax = find_minmax_t_to_source(sources_receivers_times,
                                                    i_s,
                                                    n_receivers_multiple_vector_dim,
                                                    n_receivers);

    alignas(sizeof(__m512d)) double v_buffer[vector_dim];

    _mm512_storeu_pd(v_buffer, min_elements_v);
    double min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.first
    );

    _mm512_storeu_pd(v_buffer, max_elements_v);
    double max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.second
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<float, float>
find_minmax_t_to_source_stride(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 16;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return find_minmax_t_to_source(sources_receivers_times,
                                       i_s,
                                       0,
                                       n_receivers);
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto x_stride = sources_receivers_times.get_x_stride();

    const __m512i vindex = _mm512_set_epi32(x_stride * 15,
                                            x_stride * 14,
                                            x_stride * 13,
                                            x_stride * 12,
                                            x_stride * 11,
                                            x_stride * 10,
                                            x_stride * 9,
                                            x_stride * 8,
                                            x_stride * 7,
                                            x_stride * 6,
                                            x_stride * 5,
                                            x_stride * 4,
                                            x_stride * 3,
                                            x_stride * 2,
                                            x_stride * 1,
                                            x_stride * 0);

    auto min_elements_v = _mm512_i32gather_ps(vindex,
                                              sources_receivers_times.get(i_s, 0),
                                              sizeof(float));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm512_i32gather_ps(vindex,
                                                sources_receivers_times.get(i_s, i_r + vector_dim),
                                                sizeof(float));
        min_elements_v = _mm512_min_ps(min_elements_v, new_elements);
        max_elements_v = _mm512_max_ps(max_elements_v, new_elements);
    }

    auto remainder_minmax = find_minmax_t_to_source(sources_receivers_times,
                                                    i_s,
                                                    n_receivers_multiple_vector_dim,
                                                    n_receivers);

    alignas(sizeof(__m512)) float v_buffer[vector_dim];

    _mm512_storeu_ps(v_buffer, min_elements_v);
    float min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.first
    );

    _mm512_storeu_ps(v_buffer, max_elements_v);
    float max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.second
    );

    return std::make_pair(min_el, max_el);
}

#elif __AVX2__ // AVX2 SIMD

#include <immintrin.h>

template<>
inline std::pair<double, double>
find_minmax_t_to_source_sequential(const Array2D<double> &sources_receivers_times,
                                   std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 4;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        auto minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, 0),
                                                    sources_receivers_times.get(i_s, n_receivers));
        return std::make_pair(*(minmax_iterators.first), *(minmax_iterators.second));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto min_elements_v = _mm256_loadu_pd(sources_receivers_times.get(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm256_loadu_pd(sources_receivers_times.get(i_s, i_r + vector_dim));
        min_elements_v = _mm256_min_pd(min_elements_v, new_elements);
        max_elements_v = _mm256_max_pd(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = std::minmax_element(
            sources_receivers_times.get(i_s, n_receivers_multiple_vector_dim),
            sources_receivers_times.get(i_s, n_receivers));

    alignas(sizeof(__m256d)) double v_buffer[vector_dim];

    _mm256_storeu_pd(v_buffer, min_elements_v);
    double min_el = std::min(
            *std::min_element(v_buffer, v_buffer + vector_dim),
            *(remainder_minmax_iterators.first)
    );

    _mm256_storeu_pd(v_buffer, max_elements_v);
    double max_el = std::max(
            *std::max_element(v_buffer, v_buffer + vector_dim),
            *(remainder_minmax_iterators.second)
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<float, float>
find_minmax_t_to_source_sequential(const Array2D<float> &sources_receivers_times,
                                   std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 8;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        auto minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, 0),
                                                    sources_receivers_times.get(i_s, n_receivers));
        return std::make_pair(*(minmax_iterators.first), *(minmax_iterators.second));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto min_elements_v = _mm256_loadu_ps(sources_receivers_times.get(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm256_loadu_ps(sources_receivers_times.get(i_s, i_r + vector_dim));
        min_elements_v = _mm256_min_ps(min_elements_v, new_elements);
        max_elements_v = _mm256_max_ps(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = std::minmax_element(
            sources_receivers_times.get(i_s, n_receivers_multiple_vector_dim),
            sources_receivers_times.get(i_s, n_receivers));

    alignas(sizeof(__m256)) float v_buffer[vector_dim];

    _mm256_storeu_ps(v_buffer, min_elements_v);
    float min_el = std::min(
            *std::min_element(v_buffer, v_buffer + vector_dim),
            *(remainder_minmax_iterators.first)
    );

    _mm256_storeu_ps(v_buffer, max_elements_v);
    float max_el = std::max(
            *std::max_element(v_buffer, v_buffer + vector_dim),
            *(remainder_minmax_iterators.second)
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<double, double>
find_minmax_t_to_source_stride(const Array2D<double> &sources_receivers_times,
                                                                std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 4;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return find_minmax_t_to_source(sources_receivers_times,
                                       i_s,
                                       0,
                                       n_receivers);
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto x_stride = sources_receivers_times.get_x_stride();

    const __m256i vindex = _mm256_set_epi64x(x_stride * 3,
                                             x_stride * 2,
                                             x_stride * 1,
                                             x_stride * 0);

    auto min_elements_v = _mm256_i64gather_pd(sources_receivers_times.get(i_s, 0),
                                              vindex,
                                              sizeof(double));
//    auto min_elements_v = _mm256_set_pd(sources_receivers_times(i_s, 3),
//                                        sources_receivers_times(i_s, 2),
//                                        sources_receivers_times(i_s, 1),
//                                        sources_receivers_times(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm256_i64gather_pd(sources_receivers_times.get(i_s, i_r + vector_dim),
                                                vindex,
                                                sizeof(double));
//        auto new_elements = _mm256_set_pd(sources_receivers_times(i_s, i_r + vector_dim + 3),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 2),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 1),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 0));
        min_elements_v = _mm256_min_pd(min_elements_v, new_elements);
        max_elements_v = _mm256_max_pd(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = find_minmax_t_to_source(sources_receivers_times,
                                                              i_s,
                                                              n_receivers_multiple_vector_dim,
                                                              n_receivers);

    alignas(sizeof(__m256d)) double v_buffer[vector_dim];

    _mm256_storeu_pd(v_buffer, min_elements_v);
    double min_el = std::min(
            *std::min_element(v_buffer, v_buffer + vector_dim),
            remainder_minmax_iterators.first
    );

    _mm256_storeu_pd(v_buffer, max_elements_v);
    double max_el = std::max(
            *std::max_element(v_buffer, v_buffer + vector_dim),
            remainder_minmax_iterators.second
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<float, float>
find_minmax_t_to_source_stride(const Array2D<float> &sources_receivers_times,
                                   std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 8;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return find_minmax_t_to_source(sources_receivers_times,
                                       i_s,
                                       0,
                                       n_receivers);
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers - (n_receivers % vector_dim);

    auto x_stride = sources_receivers_times.get_x_stride();

    const __m256i vindex = _mm256_set_epi32(x_stride * 7,
                                            x_stride * 6,
                                            x_stride * 5,
                                            x_stride * 4,
                                            x_stride * 3,
                                            x_stride * 2,
                                            x_stride * 1,
                                            x_stride * 0);

    auto min_elements_v = _mm256_i32gather_ps(sources_receivers_times.get(i_s, 0),
                                              vindex,
                                              sizeof(float));

//    auto min_elements_v = _mm256_set_ps(sources_receivers_times(i_s, 7),
//                                        sources_receivers_times(i_s, 6),
//                                        sources_receivers_times(i_s, 5),
//                                        sources_receivers_times(i_s, 4),
//                                        sources_receivers_times(i_s, 3),
//                                        sources_receivers_times(i_s, 2),
//                                        sources_receivers_times(i_s, 1),
//                                        sources_receivers_times(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm256_i32gather_ps(sources_receivers_times.get(i_s, i_r + vector_dim),
                                              vindex,
                                              sizeof(float));
//        auto new_elements = _mm256_set_ps(sources_receivers_times(i_s, i_r + vector_dim + 7),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 6),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 5),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 4),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 3),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 2),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 1),
//                                          sources_receivers_times(i_s, i_r + vector_dim + 0));
        min_elements_v = _mm256_min_ps(min_elements_v, new_elements);
        max_elements_v = _mm256_max_ps(max_elements_v, new_elements);
    }

    auto remainder_minmax = find_minmax_t_to_source(sources_receivers_times,
                                                    i_s,
                                                    n_receivers_multiple_vector_dim,
                                                    n_receivers);

    alignas(sizeof(__m256)) float v_buffer[vector_dim];

    _mm256_storeu_ps(v_buffer, min_elements_v);
    float min_el = std::min(
            *std::min_element(v_buffer, v_buffer + vector_dim),
            remainder_minmax.first
    );

    _mm256_storeu_ps(v_buffer, max_elements_v);
    float max_el = std::max(
            *std::max_element(v_buffer, v_buffer + vector_dim),
            remainder_minmax.second
    );

    return std::make_pair(min_el, max_el);
}

#elif __SSE2__ // SSE2 SIMD

#include <immintrin.h>

template<>
inline std::pair<double, double>
find_minmax_t_to_source_sequential(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 2;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return std::make_pair(sources_receivers_times(i_s, 0), sources_receivers_times(i_s, 0));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m128d min_elements_v = _mm_loadu_pd(sources_receivers_times.get(i_s, 0));
    __m128d max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        __m128d new_elements = _mm_loadu_pd(sources_receivers_times.get(i_s, i_r + vector_dim));
        min_elements_v = _mm_min_pd(min_elements_v, new_elements);
        max_elements_v = _mm_max_pd(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, n_receivers_multiple_vector_dim), sources_receivers_times.get(i_s, n_receivers));

    alignas(sizeof(__m128d)) double v_buffer[vector_dim];

    _mm_storeu_pd(v_buffer, min_elements_v);
    double min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.first)
    );

    _mm_storeu_pd(v_buffer, max_elements_v);
    double max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.second)
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<float, float>
find_minmax_t_to_source_sequential(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 4;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        auto minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, 0), sources_receivers_times.get(i_s, n_receivers));
        return std::make_pair(*(minmax_iterators.first), *(minmax_iterators.second));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    auto min_elements_v = _mm_loadu_ps(sources_receivers_times.get(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm_loadu_ps(sources_receivers_times.get(i_s, i_r + vector_dim));
        min_elements_v = _mm_min_ps(min_elements_v, new_elements);
        max_elements_v = _mm_max_ps(max_elements_v, new_elements);
    }

    auto remainder_minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, n_receivers_multiple_vector_dim), sources_receivers_times.get(i_s, n_receivers));

    alignas(sizeof(__m128)) float v_buffer[vector_dim];

    _mm_storeu_ps(v_buffer, min_elements_v);
    float min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.first)
    );

    _mm_storeu_ps(v_buffer, max_elements_v);
    float max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        *(remainder_minmax_iterators.second)
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<double, double>
find_minmax_t_to_source_stride(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 2;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return std::make_pair(sources_receivers_times(i_s, 0), sources_receivers_times(i_s, 0));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m128d min_elements_v = _mm_set_pd(sources_receivers_times(i_s, 1),
                                        sources_receivers_times(i_s, 0));
    __m128d max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        __m128d new_elements = _mm_set_pd(sources_receivers_times(i_s, i_r + vector_dim + 1),
                                          sources_receivers_times(i_s, i_r + vector_dim + 0));
        min_elements_v = _mm_min_pd(min_elements_v, new_elements);
        max_elements_v = _mm_max_pd(max_elements_v, new_elements);
    }

    auto remainder_minmax = find_minmax_t_to_source(sources_receivers_times,
                                                    i_s,
                                                    n_receivers_multiple_vector_dim,
                                                    n_receivers);

    alignas(sizeof(__m128d)) double v_buffer[vector_dim];

    _mm_storeu_pd(v_buffer, min_elements_v);
    double min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.first
    );

    _mm_storeu_pd(v_buffer, max_elements_v);
    double max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.second
    );

    return std::make_pair(min_el, max_el);
}

template<>
inline std::pair<float, float>
find_minmax_t_to_source_stride(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 4;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return find_minmax_t_to_source(sources_receivers_times,
                                       i_s,
                                       0,
                                       n_receivers);
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    auto min_elements_v = _mm_set_ps(sources_receivers_times(i_s, 3),
                                     sources_receivers_times(i_s, 2),
                                     sources_receivers_times(i_s, 1),
                                     sources_receivers_times(i_s, 0));
    auto max_elements_v = min_elements_v;
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        auto new_elements = _mm_set_ps(sources_receivers_times(i_s, i_r + vector_dim + 3),
                                       sources_receivers_times(i_s, i_r + vector_dim + 2),
                                       sources_receivers_times(i_s, i_r + vector_dim + 1),
                                       sources_receivers_times(i_s, i_r + vector_dim + 0));
        min_elements_v = _mm_min_ps(min_elements_v, new_elements);
        max_elements_v = _mm_max_ps(max_elements_v, new_elements);
    }

    auto remainder_minmax = find_minmax_t_to_source(sources_receivers_times,
                                                    i_s,
                                                    n_receivers_multiple_vector_dim,
                                                    n_receivers);

    alignas(sizeof(__m128)) float v_buffer[vector_dim];

    _mm_storeu_ps(v_buffer, min_elements_v);
    float min_el = std::min(
        *std::min_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.first
    );

    _mm_storeu_ps(v_buffer, max_elements_v);
    float max_el = std::max(
        *std::max_element(v_buffer, v_buffer + vector_dim),
        remainder_minmax.second
    );

    return std::make_pair(min_el, max_el);
}

#else // Without SIMD

template<typename T>
inline std::pair<T, T>
find_minmax_t_to_source_sequential(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s) {
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();
    auto minmax_iterators = std::minmax_element(sources_receivers_times.get(i_s, 0),
                                                sources_receivers_times.get(i_s, n_receivers));
    return std::make_pair(*(minmax_iterators.first), *(minmax_iterators.second));
}

template<typename T>
inline std::pair<T, T>
find_minmax_t_to_source_stride(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s) {
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();
    return find_minmax_t_to_source(sources_receivers_times, i_s, 0, n_receivers);
}

#endif //End selection of SIMD instructions

#endif //MINMAX_T_TO_SOURCE_H
