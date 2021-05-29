#ifndef SSE4_FUNCTIONS_IMPLS_H
#define SSE4_FUNCTIONS_IMPLS_H

#include <immintrin.h>
#include <cstddef>

template <class SimdImplementation>
struct SimdFunctions;

class Sse4FunctionsImpl {

    inline __m128 set1_impl(const float val1) const {
        return _mm_set1_ps(val1);
    }

    inline __m128d set1_impl(const double val1) const {
        return _mm_set1_pd(val1);
    }

    inline __m128 loadu_impl(const float *val_p) const {
        return _mm_loadu_ps(val_p);
    }

    inline __m128d loadu_impl(const double *val_p) const {
        return _mm_loadu_pd(val_p);
    }

    template<template <class> class ArrayView>
    inline __m128d get_vector_impl(const ArrayView<double> &array_view,
                                   std::ptrdiff_t y_idx,
                                   std::ptrdiff_t x_idx,
                                   __m128i) const {
        return _mm_set_pd(array_view(y_idx, x_idx + 1),
                          array_view(y_idx, x_idx));
    }

    template<template <class> class ArrayView>
    inline __m128 get_vector_impl(const ArrayView<float> &array_view,
                                  std::ptrdiff_t y_idx,
                                  std::ptrdiff_t x_idx,
                                  __m128i) const {
        return _mm_set_ps(array_view(y_idx, x_idx + 3),
                          array_view(y_idx, x_idx + 2),
                          array_view(y_idx, x_idx + 1),
                          array_view(y_idx, x_idx));
    }

    template<class VectorType>
    inline __m128i get_vindex_impl(std::ptrdiff_t stride) const;

    inline __m128 add_impl(const __m128 v1, const __m128 v2) const {
        return _mm_add_ps(v1, v2);
    }

    inline __m128d add_impl(const __m128d v1, const __m128d v2) const {
        return _mm_add_pd(v1, v2);
    }

    inline __m128 mul_impl(const __m128 v1, const __m128 v2) const {
        return _mm_mul_ps(v1, v2);
    }

    inline __m128d mul_impl(const __m128d v1, const __m128d v2) const {
        return _mm_mul_pd(v1, v2);
    }

    inline void storeu_impl(float *mem_p, const __m128 v) const {
        _mm_storeu_ps(mem_p, v);
    }

    inline void storeu_impl(double *mem_p, const __m128d v) const {
        _mm_storeu_pd(mem_p, v);
    }

    inline __m128 cmpgt_impl(const __m128 v1, const __m128 v2) const {
        return _mm_cmpgt_ps(v1, v2);
    }

    inline __m128d cmpgt_impl(const __m128d v1, const __m128d v2) const {
        return _mm_cmpgt_pd(v1, v2);
    }

    inline auto movemask_impl(const __m128 v) const -> decltype(_mm_movemask_ps(v)) {
        return _mm_movemask_ps(v);
    }

    inline auto movemask_impl(const __m128d v) const -> decltype(_mm_movemask_pd(v)) {
        return _mm_movemask_pd(v);
    }

    friend SimdFunctions<Sse4FunctionsImpl>;
};

template<>
inline __m128i Sse4FunctionsImpl::get_vindex_impl<__m128>(std::ptrdiff_t stride) const {
    return _mm_set_epi32(stride * 3,
                         stride * 2,
                         stride * 1,
                         0);
}

template<>
inline __m128i Sse4FunctionsImpl::get_vindex_impl<__m128d>(std::ptrdiff_t stride) const {
    return _mm_set_epi64x(stride * 1,
                         0);
}

#endif //SSE4_FUNCTIONS_IMPLS_H
