#ifndef SIMD_FUNCTIONS_IMPLS_H
#define SIMD_FUNCTIONS_IMPLS_H

#include <cstddef>

template<class SimdImplementation>
struct SimdFunctions {

private:
    SimdImplementation simd_implementation_;

public:

    explicit SimdFunctions(SimdImplementation simd_implementation) : simd_implementation_(simd_implementation) {}

    template<class FpType>
    inline auto
    set1(const FpType val) const -> decltype(simd_implementation_.set1_impl(val)) {
        return simd_implementation_.set1_impl(val);
    }

    template<class FpType>
    inline auto
    loadu(const FpType *val_p) const -> decltype(simd_implementation_.loadu_impl(val_p)) {
        return simd_implementation_.loadu_impl(val_p);
    }

    template<class ArrayView, class VindexType>
    inline auto
    get_vector(const ArrayView &array_view,
               std::ptrdiff_t y_idx,
               std::ptrdiff_t x_idx,
               VindexType vindex) const -> decltype(simd_implementation_.get_vector_impl(array_view, y_idx, x_idx, vindex)) {
        return simd_implementation_.get_vector_impl(array_view, y_idx, x_idx, vindex);
    }

    template<class VectorType>
    inline auto
    get_vindex(std::ptrdiff_t stride) const -> decltype(simd_implementation_.template get_vindex_impl<VectorType>(stride)) {
        return simd_implementation_.template get_vindex_impl<VectorType>(stride);
    }

    template<class VectorType>
    inline VectorType
    add(const VectorType v1, const VectorType v2) const {
        return simd_implementation_.add_impl(v1, v2);
    }

    template<class VectorType>
    inline VectorType
    mul(const VectorType v1, const VectorType v2) const {
        return simd_implementation_.mul_impl(v1, v2);
    }

    template<class FpType, class VectorType>
    inline void
    storeu(FpType *mem_p, const VectorType v) const {
        simd_implementation_.storeu_impl(mem_p, v);
    }

    template<class VectorType>
    inline auto
    cmpgt(const VectorType v1, const VectorType v2) const -> decltype(simd_implementation_.cmpgt_impl(v1, v2)) {
        return simd_implementation_.cmpgt_impl(v1, v2);
    }

    template<class VectorType>
    inline auto
    movemask(const VectorType v) const -> decltype(simd_implementation_.movemask_impl(v)) {
        return simd_implementation_.movemask_impl(v);
    }
};

#endif //SIMD_FUNCTIONS_IMPLS_H
