#ifndef _AMPLITUDES_CALCULATOR_NON_VECTORS_H
#define _AMPLITUDES_CALCULATOR_NON_VECTORS_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <type_traits>

template<typename T,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
class AmplitudesCalculatorNonVectors : public AmplitudesCalculatorBase<T, AmplitudesCalculatorNonVectors<T>> {
public:

    using value_type = T;
    using size_type = std::ptrdiff_t;

    AmplitudesCalculatorNonVectors(const Array2D<value_type> &sources_coords,
                                   const Array1D<value_type> &tensor_matrix) :
            AmplitudesCalculatorBase(sources_coords, tensor_matrix) {}

    friend AmplitudesCalculatorBase<T, AmplitudesCalculatorNonVectors<T>>;

private:

    void realize_calculate(const Array2D<value_type> &rec_coords_, Array2D<value_type> &amplitudes_) {
        this->non_vector_calculate_amplitudes(0, sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
    }
};

#endif //_AMPLITUDES_CALCULATOR_NON_VECTORS_H