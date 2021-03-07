#ifndef _TEST_DATA_GENERATE_H
#define _TEST_DATA_GENERATE_H

#include <cstddef>

class test_data_generator {

public:
	test_data_generator(double x0_r, double x1_r, std::size_t NxR,
					double y0_r, double y1_r, std::size_t NyR,
					double x0_s, double x1_s, std::size_t NxS,
					double y0_s, double y1_s, std::size_t NyS,
					double z0, double z1, std::size_t Nz,
					std::size_t n_samples, double velocity);

	~test_data_generator();

	std::size_t get_n_receivers() const;

	std::size_t get_n_sources() const;

	std::size_t get_n_samples() const;

	double get_dt() const;

	double *get_receivers_coords();

	double *get_sources_coords();

	double *get_sources_receivers_times();

	double *get_gather();

private:

	double euqlidean_dist(double x0, double x1, double y0, double y1, double z0, double z1);

	std::size_t n_receivers_;
	std::size_t n_sources_;
	std::size_t n_samples_;
	double dt_;
	double *receivers_coords_ = nullptr;
	double *sources_coords_ = nullptr;
	double *sources_receivers_times_ = nullptr;
	double *gather_ = nullptr;

};

#endif //_TEST_DATA_GENERATE_H