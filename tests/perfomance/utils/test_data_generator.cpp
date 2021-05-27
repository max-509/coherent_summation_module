#include "test_data_generator.h"

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

test_data_generator::test_data_generator(double x0_r, double x1_r, std::size_t NxR,
										double y0_r, double y1_r, std::size_t NyR,
										double x0_s, double x1_s, std::size_t NxS,
										double y0_s, double y1_s, std::size_t NyS,
										double z0, double z1, std::size_t Nz,
										std::size_t n_samples, double velocity) : n_receivers_(NxR*NyR),
																					n_sources_(NxS*NyS*Nz),
																					n_samples_(n_samples),
																					receivers_coords_(new double[n_receivers_*3]), 
																				 	sources_coords_(new double[n_sources_*3]),
																				 	sources_receivers_times_(new double[n_receivers_*n_sources_]),
																				 	gather_(new double[n_receivers_*n_samples_]) 
{
	double hx_r = (x1_r - x0_r) / (NxR - 1);
	double hy_r = (y1_r = y0_r) / (NyR - 1);
	
	for (std::size_t iy_r = 0; iy_r < NyR; ++iy_r) {
		double y_coord = y0_r + hy_r*iy_r;	
		for (std::size_t ix_r = 0; ix_r < NxR; ++ix_r) {
			receivers_coords_[3*(iy_r*NxR + ix_r) + 0] = x0_r + hx_r*ix_r;
			receivers_coords_[3*(iy_r*NxR + ix_r) + 1] = y_coord;
			receivers_coords_[3*(iy_r*NxR + ix_r) + 2] = 0.0;
		}
	}

	double hx_s = (x1_s - x0_s) / (NxS - 1);
	double hy_s = (y1_s - y0_s) / (NyS - 1);
	double hz_s = (z0 - z1) / (Nz - 1);
	
	for (std::size_t iz_s = 0; iz_s < Nz; ++iz_s) {
		double z_coord = z0 + hz_s*iz_s;
		for (std::size_t iy_s = 0; iy_s < NyS; ++iy_s) {
			double y_coord = y0_s + iy_s*hy_s;
			for (std::size_t ix_s = 0; ix_s < NxS; ++ix_s) {
				sources_coords_[3*(iz_s*NyS*NxS + iy_s*NxS + ix_s) + 0] = x0_s + hx_s*ix_s;
				sources_coords_[3*(iz_s*NyS*NxS + iy_s*NxS + ix_s) + 1] = y_coord;
				sources_coords_[3*(iz_s*NyS*NxS + iy_s*NxS + ix_s) + 2] = z_coord;
			}	
		}
	}

	
	for (std::size_t i_s = 0; i_s < n_sources_; ++i_s) {
		double x_s = sources_coords_[i_s*3 + 0];
		double y_s = sources_coords_[i_s*3 + 1];
		double z_s = sources_coords_[i_s*3 + 2];
		for (std::size_t i_r = 0; i_r < n_receivers_; ++i_r) {
			double x_r = receivers_coords_[i_r*3 + 0];
			double y_r = receivers_coords_[i_r*3 + 1];
			double z_r = receivers_coords_[i_r*3 + 2];
			sources_receivers_times_[i_s*n_receivers_ + i_r] = euqlidean_dist(x_r, x_s, y_r, y_s, z_r, z_s) / velocity;
		}
	}

	auto minmax_t_it = std::minmax_element(sources_receivers_times_, sources_receivers_times_ + n_sources_*n_receivers_);
	double min_t = *minmax_t_it.first;
	double max_t = *minmax_t_it.second;

	dt_ = (max_t - min_t) / (n_samples_ - 1);

	srand(time(nullptr));

	
	for (std::size_t i_r = 0; i_r < n_receivers_; ++i_r) {
		for (std::size_t i_n = 0; i_n < n_samples_; ++i_n) {
			gather_[i_r*n_samples_ + i_n] = (double)rand() / RAND_MAX;
		}
	}

}

test_data_generator::~test_data_generator() {
	delete [] gather_;
	delete [] receivers_coords_;
	delete [] sources_coords_;
	delete [] sources_receivers_times_;
}

std::size_t test_data_generator::get_n_receivers() const {
	return n_receivers_;
}

std::size_t test_data_generator::get_n_sources() const {
	return n_sources_;
}

std::size_t test_data_generator::get_n_samples() const {
	return n_samples_;
}

double test_data_generator::get_dt() const {
	return dt_;
}

double *test_data_generator::get_receivers_coords() {
	return receivers_coords_;
}

double *test_data_generator::get_sources_coords() {
	return sources_coords_;
}

double *test_data_generator::get_sources_receivers_times() {
	return sources_receivers_times_;
}

double *test_data_generator::get_gather() {
	return gather_;
}

double test_data_generator::euqlidean_dist(double x0, double x1, double y0, double y1, double z0, double z1) {
	return std::sqrt(std::pow(x1-x0, 2) + std::pow(y1-y0, 2) + std::pow(z1-z0, 2));
}