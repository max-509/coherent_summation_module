#include "test_data_generator2D.h"

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <initializer_list>

test_data_generator2D::test_data_generator2D(double x0_r, double x1_r, std::size_t NxR,
										double x0_s, double x1_s, std::size_t NxS,
										double z0, double z1, std::size_t Nz,
										double s_x,
										double dt,
										std::size_t n_samples, 
										std::vector<double> velocities,
										std::vector<double> borders) : n_receivers_(NxR),
																		x_dim_(NxS),
																		z_dim_(Nz),
																		n_samples_(n_samples),
																		s_x_(s_x),
																		dt_(dt),
																		receivers_coords_(n_receivers_),
																	 	velocity_model_(new double[x_dim_*z_dim_]),
																	 	grid_({std::make_pair(z0, z1), std::make_pair(x0_s, x1_s)}),
																	 	gather_(new double[n_receivers_*n_samples_]) 
{
	double hx_r = (x1_r - x0_r) / (NxR - 1);

	for (std::size_t ix_r = 0; ix_r < NxR; ++ix_r) {
		receivers_coords_[ix_r] = x0_r + hx_r*ix_r;
	}

	double dz = (z1 - z0) / (z_dim_ - 1);

	for (std::size_t i_z = 0; i_z < z_dim_; ++i_z) {
		double z = z0 + dz*i_z;

		double border = 0;
		double vel_in_layer = velocities.back();
		for (std::size_t i_b = 0; i_b < borders.size(); ++i_b) {
			border += borders[i_b];
			if (z <= border) {
				vel_in_layer = velocities[i_b];
				break;
			}
		}

		for (std::size_t i_x = 0; i_x < x_dim_; ++i_x) {
			velocity_model_[i_z*NxS + i_x] = vel_in_layer;
		}
	}

	srand(time(nullptr));
	
	for (std::size_t i_r = 0; i_r < n_receivers_; ++i_r) {
		for (std::size_t i_n = 0; i_n < n_samples_; ++i_n) {
			gather_[i_r*n_samples_ + i_n] = (double)rand() / RAND_MAX;
		}
	}

}

test_data_generator2D::~test_data_generator2D() {
	delete [] gather_;
	delete [] velocity_model_;
}

std::size_t test_data_generator2D::get_n_receivers() const {
	return n_receivers_;
}

std::size_t test_data_generator2D::get_x_dim() const {
	return x_dim_;
}

std::size_t test_data_generator2D::get_z_dim() const {
	return z_dim_;
}

std::size_t test_data_generator2D::get_n_samples() const {
	return n_samples_;
}

double test_data_generator2D::get_s_x() const {
	return s_x_;
}

double test_data_generator2D::get_dt() const {
	return dt_;
}

std::vector<double> &test_data_generator2D::get_receivers_coords() {
	return receivers_coords_;
}

double *test_data_generator2D::get_velocity_model() {
	return velocity_model_;
}

std::vector<std::pair<double, double>> &test_data_generator2D::get_grid() {
	return grid_;
}

double *test_data_generator2D::get_gather() {
	return gather_;
}