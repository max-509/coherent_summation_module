#ifndef _TEST_DATA_GENERATE2D_H
#define _TEST_DATA_GENERATE2D_H

#include <cstddef>
#include <tuple>
#include <vector>

class test_data_generator2D {

public:
	test_data_generator2D(double x0_r, double x1_r, std::size_t NxR,
					double x0_s, double x1_s, std::size_t NxS,
					double z0, double z1, std::size_t Nz,
					double s_x,
					double dt,
					std::size_t n_samples, 
					std::vector<double> velocities,
					std::vector<double> borders);

	~test_data_generator2D();

	std::size_t get_n_receivers() const;

	std::size_t get_x_dim() const;

	std::size_t get_z_dim() const;

	std::size_t get_n_samples() const;

	double get_s_x() const;

	double get_dt() const;

	std::vector<double> &get_receivers_coords();

	double *get_velocity_model();

	std::vector<std::pair<double, double>> &get_grid();

	double *get_gather();

private:

	std::size_t n_receivers_;
	std::size_t x_dim_;
	std::size_t z_dim_;
	std::size_t n_samples_;
	double s_x_;
	double dt_;
	std::vector<double> receivers_coords_;
	double *velocity_model_ = nullptr;
	std::vector<std::pair<double, double>> grid_;
	double *gather_ = nullptr;

};

#endif //_TEST_DATA_GENERATE2D_H