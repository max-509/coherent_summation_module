#ifndef _ARRIVAL_TIMES_CALCULATOR_3D_H
#define _ARRIVAL_TIMES_CALCULATOR_3D_H

#include <vector>
#include <cstddef>
#include <tuple>

template <typename CalculatorType, typename T>
class ArrivalTimesCalculator3D {
public:
	ArrivalTimesCalculator3D(const Array3D<T> &velocity_model,
							const std::vector<std::pair<double, double>> &grid) : 
									velocity_model_(velocity_model),
									grid_(grid),
									calculator_(velocity_model_, grid_)
	{ }

	const std::vector<std::pair<double, double>> &get_grid() const {
		return grid_;
	}

	std::ptrdiff_t get_z_dim() const {
		return velocity_model_.get_z_dim();
	}

	std::ptrdiff_t get_y_dim() const {
		return velocity_model_.get_y_dim();
	}

	std::ptrdiff_t get_x_dim() const {
		return velocity_model_.get_x_dim();
	}

	double operator()(double x, double y, double p_z, double p_y, double p_x) {
		return calculator_.calculate(x, y, p_z, p_y, p_x);
	}


	std::vector<double> operator()(std::vector<double> &x,
									std::vector<double> &y,
									std::vector<double> &p_z,
									std::vector<double> &p_y,
									std::vector<double> &p_x) {
		return calculator_.calculate(x, y, p_z, p_y, p_x);
	}

private:
	const Array3D<T> &velocity_model_;
	const std::vector<std::pair<double, double>> &grid_;
	CalculatorType calculator_;
};

#endif //_ARRIVAL_TIMES_CALCULATOR_3D_H