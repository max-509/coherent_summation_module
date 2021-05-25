#ifndef _AMPLITUDES_CALCULATOR_BASE_H
#define _AMPLITUDES_CALCULATOR_BASE_H

#include "array2D.h"

#include <limits>
#include <cstddef>
#include <cmath>
#include <type_traits>

template <typename InputArrayType,
        typename Realization,
        typename std::enable_if<std::is_floating_point<typename InputArrayType::value_type>::value, bool>::type = true>
class AmplitudesCalculatorBase {
public:

    using value_type = typename std::remove_const<typename InputArrayType::value_type>::type;
    using size_type = typename InputArrayType::size_type;

    template<typename OutputArrayType>
	void calculate(const InputArrayType &rec_coords, OutputArrayType &amplitudes) {
		static_cast<Realization*>(this)->realize_calculate(rec_coords, amplitudes); 
	}

	friend Realization;

private:

    template <typename OutputArrayType>
	void non_vector_calculate_amplitudes(size_type ind_first_rec, const InputArrayType &sources_coords, const InputArrayType &rec_coords, const value_type * tensor_matrix, OutputArrayType &amplitudes) {
	    size_type n_rec = rec_coords.get_y_dim();
	    size_type sources_count = sources_coords.get_y_dim();
	    constexpr size_type matrix_size = size_type(6);

	    constexpr value_type two_T = value_type(2.0);

	    value_type coord_vect[3];
	    value_type G_P[matrix_size];

	    #ifdef _MSC_VER
	    #pragma omp parallel for schedule(static) collapse(2) private(coord_vect, G_P)
        #else //_MSC_VER
	    #pragma omp parallel for simd schedule(static) collapse(2) private(coord_vect, G_P)
	    #endif //_MSC_VER
        for (size_type i = size_type(0); i < sources_count; ++i) {
            for (size_type r_ind = ind_first_rec; r_ind < n_rec; ++r_ind) {
                for (size_type crd = size_type(0); crd < size_type(3); ++crd) {
                    coord_vect[crd] = rec_coords(r_ind, crd) - sources_coords(i, crd);
                }

                value_type rev_dist = value_type(1.0) / calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);
                for (size_type crd = size_type(2); crd >= size_type(0); --crd) {
                    coord_vect[crd] *= rev_dist;
                    G_P[crd] = coord_vect[2] * coord_vect[crd] * coord_vect[crd] * rev_dist;
                }

                value_type double_norm_coord_z = two_T * coord_vect[2] * rev_dist;

                G_P[3] = double_norm_coord_z * coord_vect[1] * coord_vect[2];
                G_P[4] = double_norm_coord_z * coord_vect[0] * coord_vect[2];
                G_P[5] = double_norm_coord_z * coord_vect[0] * coord_vect[1];


                value_type ampl_tmp = 0.0;
                for (size_type m = size_type(0); m < matrix_size; ++m) {
                    ampl_tmp += (G_P[m]) * tensor_matrix[m];
                }
                ampl_tmp /= (std::fabs(ampl_tmp) + std::numeric_limits<value_type>::epsilon());

                amplitudes(i, r_ind) = ampl_tmp;
            }
        }

	}

	inline value_type calc_norm(value_type x, value_type y, value_type z) {
	    using namespace std;
	    return sqrt(x*x+y*y+z*z)+numeric_limits<value_type>::epsilon();
	}
};

#endif //_AMPLITUDES_CALCULATOR_BASE_H