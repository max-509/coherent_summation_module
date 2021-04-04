#ifndef _AMPLITUDES_CALCULATOR_NON_VECTORS_H
#define _AMPLITUDES_CALCULATOR_NON_VECTORS_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

template <typename T>
class AmplitudesCalculatorNonVectors : public AmplitudesCalculatorBase<T, AmplitudesCalculatorNonVectors<T>> {
public:
	AmplitudesCalculatorNonVectors(const Array2D<T> &sources_coords,
						 	  	  const T * tensor_matrix) :
		sources_coords_(sources_coords),
		tensor_matrix_(tensor_matrix)
	{ }

	friend AmplitudesCalculatorBase<T, AmplitudesCalculatorNonVectors<T>>;

private:
	const Array2D<T> &sources_coords_;
	const T * tensor_matrix_;

	void realize_calculate(const Array2D<T> &rec_coords_, Array2D<T> &amplitudes_) {
		this->non_vector_calculate_amplitudes(0, sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
	}
};

#endif //_AMPLITUDES_CALCULATOR_NON_VECTORS_H