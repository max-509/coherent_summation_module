#ifndef AVX512_FUNCTIONS_IMPLS_H
#define AVX512_FUNCTIONS_IMPLS_H

#include <immintrin.h>

template <class SimdImplementation>
struct SimdFunctions;

class Avx512FunctionsImpl {

    inline __m512 set1_impl(const float val1) const {
        return _mm512_set1_ps(val1);
    }

    inline __m512d set1_impl(const double val1) const {
        return _mm512_set1_pd(val1);
    }

    inline __m512 loadu_impl(const float *val_p) const {
        return _mm512_loadu_ps(val_p);
    }

    inline __m512d loadu_impl(const double *val_p) const {
        return _mm512_loadu_pd(val_p);
    }

    template<template <class> class ArrayView, typename IdxType>
    inline __m512d get_vector_impl(const ArrayView<double> &array_view,
                                   IdxType y_idx,
                                   IdxType x_idx,
                                   __m512i vindex) const {
        return _mm512_i64gather_pd(vindex,
                                   array_view.get(y_idx, x_idx),
                                   sizeof(double));
    }

    template<template <class> class ArrayView, typename IdxType>
    inline __m512 get_vector_impl(const ArrayView<float> &array_view,
                                  IdxType y_idx,
                                  IdxType x_idx,
                                  __m512i vindex) const {
        return _mm512_i32gather_ps(vindex,
                                   array_view.get(y_idx, x_idx),
                                   sizeof(float));
    }

    template<class VectorType>
    inline __m256i get_vindex_impl(std::ptrdiff_t stride) const;

    inline __m512 add_impl(const __m512 v1, const __m512 v2) const {
        return _mm512_add_ps(v1, v2);
    }

    inline __m512d add_impl(const __m512d v1, const __m512d v2) const {
        return _mm512_add_pd(v1, v2);
    }

    inline __m512 mul_impl(const __m512 v1, const __m512 v2) const {
        return _mm512_mul_ps(v1, v2);
    }

    inline __m512d mul_impl(const __m512d v1, const __m512d v2) const {
        return _mm512_mul_pd(v1, v2);
    }

    inline void storeu_impl(float *mem_p, const __m512 v) const {
        _mm512_storeu_ps(mem_p, v);
    }

    inline void storeu_impl(double *mem_p, const __m512d v) const {
        _mm512_storeu_pd(mem_p, v);
    }

    inline auto cmpgt_impl(const __m512 v1, const __m512 v2) const-> decltype(_mm512_cmp_ps_mask(v1, v2, _CMP_GT_OQ)) {
        return _mm512_cmp_ps_mask(v1, v2, _CMP_GT_OQ);
    }

    inline auto cmpgt_impl(const __m512d v1, const __m512d v2) const -> decltype(_mm512_cmp_pd_mask(v1, v2, _CMP_GT_OQ)) {
        return _mm512_cmp_pd_mask(v1, v2, _CMP_GT_OQ);
    }

    inline auto movemask_impl(const __mmask16 v) const -> decltype(v) {
        return v;
    }

    inline auto movemask_impl(const __mmask8 v) const -> decltype(v) {
        return v;
    }

    friend SimdFunctions<Avx512FunctionsImpl>;
};

template<>
inline __m512i Avx512FunctionsImpl::get_vindex_impl<__m512>(std::ptrdiff_t stride) const {
    return _mm512_set_epi32(stride * 15,
                            stride * 14,
                            stride * 13,
                            stride * 12,
                            stride * 11,
                            stride * 10,
                            stride * 9,
                            stride * 8,
                            stride * 7,
                            stride * 6,
                            stride * 5,
                            stride * 4,
                            stride * 3,
                            stride * 2,
                            stride * 1,
                            0);
}

template<>
inline __m512i Avx512FunctionsImpl::get_vindex_impl<__m512d>(std::ptrdiff_t stride) const {
    return _mm512_set_epi64(stride * 7,
                            stride * 6,
                            stride * 5,
                            stride * 4,
                            stride * 3,
                            stride * 2,
                            stride * 1,
                            0);
}

#endif //AVX512_FUNCTIONS_IMPLS_H

