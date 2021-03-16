#ifndef _TEST_DATA_GENERATE3D_H
#define _TEST_DATA_GENERATE3D_H

#include <cstddef>
#include <tuple>
#include <vector>

class test_data_generator3D {

public:
	test_data_generator3D(double x0_r, double x1_r, std::size_t NxR,
						double y0_r, double y1_r, std::size_t NyR,
						double x0_s, double x1_s, std::size_t NxS,
						double y0_s, double y1_s, std::size_t NyS,
						double z0, double z1, std::size_t Nz,
						double s_x, double s_y,
						double dt,
						std::size_t n_samples, 
						std::vector<double> velocities,
						std::vector<double> borders);

	~test_data_generator3D();

	std::size_t get_n_receivers() const;

	std::size_t get_x_dim() const;

	std::size_t get_y_dim() const;

	std::size_t get_z_dim() const;

	std::size_t get_n_samples() const;

	double get_s_x() const;

	double get_s_y() const;

	double get_dt() const;

	double *get_receivers_coords();

	double *get_velocity_model();

	std::vector<std::pair<double, double>> &get_grid();

	double *get_gather();

private:

	std::size_t n_receivers_;
	std::size_t x_dim_;
	std::size_t y_dim_;
	std::size_t z_dim_;
	std::size_t n_samples_;
	double s_x_;
	double s_y_;
	double dt_;
	double *receivers_coords_ = nullptr;
	double *velocity_model_ = nullptr;
	std::vector<std::pair<double, double>> grid_;
	double *gather_ = nullptr;

};

#endif //_TEST_DATA_GENERATE3D_H