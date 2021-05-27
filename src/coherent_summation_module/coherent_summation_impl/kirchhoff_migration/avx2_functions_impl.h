#ifndef AVX2_FUNCTIONS_IMPLS_H
#define AVX2_FUNCTIONS_IMPLS_H

#include <immintrin.h>

template<class SimdImplementation>
struct SimdFunctions;

class Avx2FunctionsImpl {

    inline __m256 set1_impl(const float val1) const {
        return _mm256_set1_ps(val1);
    }

    inline __m256d set1_impl(const double val1) const {
        return _mm256_set1_pd(val1);
    }

    inline __m256 loadu_impl(const float *val_p) const {
        return _mm256_loadu_ps(val_p);
    }

    inline __m256d loadu_impl(const double *val_p) const {
        return _mm256_loadu_pd(val_p);
    }

    template<template<class> class ArrayView, typename IdxType>
    inline __m256d get_vector_impl(const ArrayView<double> &array_view,
                                   IdxType y_idx,
                                   IdxType x_idx,
                                   __m256i vindex) const {
        return _mm256_i64gather_pd(array_view.get(y_idx, x_idx),
                                   vindex,
                                   sizeof(double));
    }

    template<template<class> class ArrayView, typename IdxType>
    inline __m256 get_vector_impl(const ArrayView<float> &array_view,
                                  IdxType y_idx,
                                  IdxType x_idx,
                                  __m256i vindex) const {
        return _mm256_i32gather_ps(array_view.get(y_idx, x_idx),
                                   vindex,
                                   sizeof(float));
    }

    template<class VectorType>
    inline __m256i get_vindex_impl(std::ptrdiff_t stride) const;

    template<>
    inline __m256i get_vindex_impl<__m256>(std::ptrdiff_t stride) const {
        return _mm256_set_epi32(stride * 7,
                                stride * 6,
                                stride * 5,
                                stride * 4,
                                stride * 3,
                                stride * 2,
                                stride * 1,
                                0);
    }

    template<>
    inline __m256i get_vindex_impl<__m256d>(std::ptrdiff_t stride) const {
        return _mm256_set_epi64x(stride * 3,
                                 stride * 2,
                                 stride * 1,
                                 0);
    }

    inline __m256 add_impl(const __m256 v1, const __m256 v2) const {
        return _mm256_add_ps(v1, v2);
    }

    inline __m256d add_impl(const __m256d v1, const __m256d v2) const {
        return _mm256_add_pd(v1, v2);
    }

    inline __m256 mul_impl(const __m256 v1, const __m256 v2) const {
        return _mm256_mul_ps(v1, v2);
    }

    inline __m256d mul_impl(const __m256d v1, const __m256d v2) const {
        return _mm256_mul_pd(v1, v2);
    }

    inline void storeu_impl(float *mem_p, const __m256 v) const {
        _mm256_storeu_ps(mem_p, v);
    }

    inline void storeu_impl(double *mem_p, const __m256d v) const {
        _mm256_storeu_pd(mem_p, v);
    }

    inline __m256 cmpgt_impl(const __m256 v1, const __m256 v2) const {
        return _mm256_cmp_ps(v1, v2, _CMP_GT_OQ);
    }

    inline __m256d cmpgt_impl(const __m256d v1, const __m256d v2) const {
        return _mm256_cmp_pd(v1, v2, _CMP_GT_OQ);
    }

    inline auto movemask_impl(const __m256 v) const -> decltype(_mm256_movemask_ps(v)) {
        return _mm256_movemask_ps(v);
    }

    inline auto movemask_impl(const __m256d v) const -> decltype(_mm256_movemask_pd(v)) {
        return _mm256_movemask_pd(v);
    }

    friend SimdFunctions<Avx2FunctionsImpl>;
};

#endif //AVX2_FUNCTIONS_IMPLS_H
