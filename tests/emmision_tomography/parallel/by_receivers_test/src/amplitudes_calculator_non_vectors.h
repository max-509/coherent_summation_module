#ifndef _AMPLITUDES_CALCULATOR_NON_VECTORS_H
#define _AMPLITUDES_CALCULATOR_NON_VECTORS_H

#include "amplitudes_calculator_base.h"
#include "array2D.h"

template <typename T>
class AmplitudesCalculatorNonVectors : public AmplitudesCalculatorBase<T, AmplitudesCalculatorNonVectors<T>> {
public:
	AmplitudesCalculatorNonVectors(const Array2D<T> &sources_coords,
						 	  	  const Array2D<T> &rec_coords,
						 	  	  const T *tensor_matrix,
						 	  	  Array2D<T> &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculatorBase<T, AmplitudesCalculatorNonVectors<T>>;

private:
	const Array2D<T> &sources_coords_;
	const Array2D<T> &rec_coords_;
	const T *tensor_matrix_;
	Array2D<T> &amplitudes_;	

	void realize_calculate() {
		this->non_vector_calculate_amplitudes(0, sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
	}
};

#endif //_AMPLITUDES_CALCULATOR_NON_VECTORS_H