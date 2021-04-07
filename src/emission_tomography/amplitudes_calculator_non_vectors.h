#ifndef _AMPLITUDES_CALCULATOR_NON_VECTORS_H
#define _AMPLITUDES_CALCULATOR_NON_VECTORS_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

#include <type_traits>

template <typename InputArrayType,
        typename std::enable_if<std::is_floating_point<typename InputArrayType::value_type>::value, bool>::type = true>
class AmplitudesCalculatorNonVectors : public AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorNonVectors<InputArrayType>> {
public:

    using typename AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorNonVectors<InputArrayType>>::value_type;
    using typename AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorNonVectors<InputArrayType>>::size_type;

	AmplitudesCalculatorNonVectors(InputArrayType &sources_coords,
						 	  	  const value_type *tensor_matrix) :
		sources_coords_(sources_coords),
		tensor_matrix_(tensor_matrix)
	{ }

	friend AmplitudesCalculatorBase<InputArrayType, AmplitudesCalculatorNonVectors<InputArrayType>>;

private:
	InputArrayType &sources_coords_;
	const value_type *tensor_matrix_;

	template <typename OutputArrayType>
	void realize_calculate(InputArrayType &rec_coords_, OutputArrayType &amplitudes_) {
		this->non_vector_calculate_amplitudes(0, sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
	}
};

#endif //_AMPLITUDES_CALCULATOR_NON_VECTORS_H