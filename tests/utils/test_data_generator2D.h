#ifndef _TEST_DATA_GENERATE2D_H
#define _TEST_DATA_GENERATE2D_H

#include <cstddef>
#include <tuple>
#include <vector>
#include <memory>

template <typename T2>
class test_data_generator2D {

public:
	test_data_generator2D(double x0_s, double x1_s, std::size_t NxS,
                       double z0, double z1, std::size_t Nz,
                       double s_x,
                       double dt,
                       std::size_t n_samples,
                       T2 velocity)
                       : x_dim_(NxS),
                       z_dim_(Nz),
                       n_samples_(n_samples),
                       dt_(dt),
                       velocity_(velocity),
                       times_to_source_(z_dim_*x_dim_),
                       grid_{std::make_pair(z0, z1), std::make_pair(x0_s, x1_s)}
{

	double hx_s = (x1_s - x0_s) / (NxS - 1);
	double hz_s = (z0 - z1) / (Nz - 1);

	for (std::size_t i_z = 0; i_z < z_dim_; ++i_z) {
	    double p_z = z0 + hz_s*i_z;
	    for (std::size_t i_x = 0; i_x < x_dim_; ++i_x) {
	        double p_x = x0_s + hx_s*i_x;

	        std::size_t i_p = i_z*x_dim_ + i_x;

	        times_to_source_[i_p] = euqlidean_dist(s_x, p_x, 0, 0, 0, p_z);
	    }
	}

}

    template <typename T1>
    std::pair<std::unique_ptr<T2[]>, std::unique_ptr<T1[]>>
    generate_user_data_by_receivers(const std::vector<double> &receivers_coords,
                                    bool is_transpose_receivers) {
	    std::size_t n_receivers = receivers_coords.size();
	    std::unique_ptr<T2[]> times_to_receivers(new T2[z_dim_*x_dim_*n_receivers]);
	    std::unique_ptr<T1[]> gather(new T1[n_receivers * n_samples_]);

	    double z0 = grid_[0].first, z1 = grid_[0].second;
	    double x0 = grid_[1].first, x1 = grid_[1].second;
	    double dz = (z1 - z0) / (z_dim_ - 1);
	    double dx = (x1 - x0) / (x_dim_ - 1);

	    if (is_transpose_receivers) {
	        for (std::size_t i_r = 0; i_r < n_receivers; ++i_r) {
	            double r_x = receivers_coords[i_r];
	            for (std::size_t i_z = 0; i_z < z_dim_; ++i_z) {
	                double p_z = z0 + dz*i_z;
	                for (std::size_t i_x = 0; i_x < x_dim_; ++i_x) {
	                    double p_x = x0 + dx*i_x;

	                    std::size_t i_p = i_z*x_dim_ + i_x;

	                    times_to_receivers[i_r*z_dim_*x_dim_ + i_p] =
                            euqlidean_dist(r_x, p_x, 0, 0, 0, p_z) / velocity_;
	                }
	            }
	        }
	    } else {
	        for (std::size_t i_z = 0; i_z < z_dim_; ++i_z) {
                double p_z = z0 + dz*i_z;
                for (std::size_t i_x = 0; i_x < x_dim_; ++i_x) {
                    double p_x = x0 + dx*i_x;

                    std::size_t i_p = i_z*x_dim_ + i_x;
                    for (std::size_t i_r = 0; i_r < n_receivers; ++i_r) {
                        double r_x = receivers_coords[i_r];

                        times_to_receivers[i_p * n_receivers + i_r] =
                            euqlidean_dist(r_x, p_x, 0, 0, 0, p_z) / velocity_;
                    }
                }
            }
	    }

	    srand(time(nullptr));

    	for (std::size_t i_r = 0; i_r < n_receivers; ++i_r) {
    		for (std::size_t i_n = 0; i_n < n_samples_; ++i_n) {
    			gather[i_r*n_samples_ + i_n] = (double)rand() / RAND_MAX;
    		}
    	}

        return std::make_pair(std::move(times_to_receivers), std::move(gather));
	}

    std::size_t get_x_dim() const {
        return x_dim_;
    }

    std::size_t get_z_dim() const {
        return z_dim_;
    }

    std::size_t get_n_samples() const {
        return n_samples_;
    }

    double get_dt() const {
        return dt_;
    }

    std::vector<T2> &get_times_to_source() {
        return times_to_source_;
    }

private:

	std::size_t x_dim_;
	std::size_t z_dim_;
	std::size_t n_samples_;
	double dt_;
	T2 velocity_;
	std::vector<T2> times_to_source_;
	std::vector<std::pair<double, double>> grid_;

	T2 euqlidean_dist(double x0, double x1, double y0, double y1, double z0, double z1) {
        return std::sqrt(std::pow(x1-x0, 2) + std::pow(y1-y0, 2) + std::pow(z1-z0, 2));
    }

};

#endif //_TEST_DATA_GENERATE2D_H