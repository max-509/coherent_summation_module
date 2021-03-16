#ifndef ARRIVAL_TIMES_BY_HORIZONTALLY_LAYERED_MEDIUM_2D_H
#define ARRIVAL_TIMES_BY_HORIZONTALLY_LAYERED_MEDIUM_2D_H

#include "array2D.h"

#include <cstddef>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vector>
#include <tuple>

template <typename T>
class ArrivalTimesByHorizontallyLayeredMedium2D {
public:
	ArrivalTimesByHorizontallyLayeredMedium2D(const Array2D<T> &velocity_model,
											const std::vector<std::pair<double, double>> &grid) : v_rms(0.0) {
		std::ptrdiff_t x_dim =  velocity_model.get_x_dim(), z_dim = velocity_model.get_y_dim();

		double z0 = grid[0].first, z1 = grid[0].second;
	    double dz = z1 - z0;

	    T sum_t = 0.0;

		for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
			double v_mean_layer = std::accumulate(&velocity_model(i_z, 0), &velocity_model(i_z, x_dim), static_cast<T>(0.0)) / x_dim;
			double z = z0 + i_z*dz;
			T t_z = z / v_mean_layer;
			v_rms += (t_z * std::pow(v_mean_layer, 2));
			sum_t += t_z;
		}
		v_rms = std::sqrt(v_rms / sum_t);
		inv_v_rms = 1.0 / v_rms;
	}

	inline double calculate(double x, double p_z, double p_x) {
		return std::sqrt(p_z*p_z + std::pow(x - p_x, 2)) * inv_v_rms;
	}

	std::vector<double> calculate(std::vector<double> &x,
								  std::vector<double> &p_z,
								  std::vector<double> &p_x) {
		std::vector<double> times(x.size());
		for (std::size_t i_t = 0; i_t < times.size(); ++i_t) {
			times[i_t] = calculate(x[i_t], p_z[i_t], p_x[i_t]);
		}
		return times;
	}

private:
	T v_rms;
	T inv_v_rms;
};

#endif //ARRIVAL_TIMES_BY_HORIZONTALLY_LAYERED_MEDIUM_2D_H