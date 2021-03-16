#ifndef _ARRIVAL_TIMES_CALCULATOR_2D_H
#define _ARRIVAL_TIMES_CALCULATOR_2D_H

#include <vector>
#include <cstddef>
#include <tuple>

template <typename CalculatorType, typename T>
class ArrivalTimesCalculator2D {
public:
	ArrivalTimesCalculator2D(const Array2D<T> &velocity_model,
							const std::vector<std::pair<double, double>> &grid) : 
									velocity_model_(velocity_model),
									grid_(grid),
									calculator_(velocity_model_, grid_)
	{ }

	const std::vector<std::pair<double, double>> &get_grid() const {
		return grid_;
	}

	std::ptrdiff_t get_z_dim() const {
		return velocity_model_.get_y_dim();
	}

	std::ptrdiff_t get_x_dim() const {
		return velocity_model_.get_x_dim();
	}

	double operator()(double x, double p_z, double p_x) {
		return calculator_.calculate(x, p_z, p_x);
	}


	std::vector<double> operator()(std::vector<double> &x,
									std::vector<double> &p_z,
									std::vector<double> &p_x) {
		return calculator_.calculate(x, p_z, p_x);
	}

private:
	const Array2D<T> &velocity_model_;
	const std::vector<std::pair<double, double>> &grid_;
	CalculatorType calculator_;
};

#endif //_ARRIVAL_TIMES_CALCULATOR_2D_H