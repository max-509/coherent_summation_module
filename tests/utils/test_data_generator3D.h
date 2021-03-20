#ifndef _TEST_DATA_GENERATE3D_H
#define _TEST_DATA_GENERATE3D_H

#include <cstddef>
#include <tuple>
#include <vector>
#include <cmath>
#include <memory>

template <typename T2>
class test_data_generator3D {

public:
	test_data_generator3D(double x0_s, double x1_s, std::size_t NxS,
						double y0_s, double y1_s, std::size_t NyS,
						double z0, double z1, std::size_t Nz,
						double s_x, double s_y,
						double dt,
						std::size_t n_samples, 
						double velocity)
						: x_dim_(NxS),
						y_dim_(NyS),
						z_dim_(Nz),
						n_samples_(n_samples),
						dt_(dt),
						velocity_(velocity),
						times_to_source_(z_dim_*y_dim_*x_dim_),
						grid_{std::make_pair(z0, z1),
                                std::make_pair(y0_s, y1_s),
                                std::make_pair(x0_s, x1_s)} {

        double hx_s = (x1_s - x0_s) / (NxS - 1);
        double hy_s = (y1_s - y0_s) / (NyS - 1);
        double hz_s = (z0 - z1) / (Nz - 1);

        for (std::size_t i_p_z = 0; i_p_z < z_dim_; ++i_p_z) {
            double p_z = z0 + hz_s * i_p_z;
            for (std::size_t i_p_y = 0; i_p_y < y_dim_; ++i_p_y) {
                double p_y = y0_s + hy_s * i_p_y;
                for (std::size_t i_p_x = 0; i_p_x < x_dim_; ++i_p_x) {
                    double p_x = x0_s + hx_s * i_p_x;

                    std::size_t i_p = (i_p_z * y_dim_ + i_p_y) * x_dim_ + i_p_x;
                    times_to_source_[i_p] = euqlidean_dist(s_x, p_x, s_y, p_y, 0, p_z);

//                        for (std::size_t i_r_y = 0; i_r_y < NyR; ++i_r_y) {
//                            double r_y = y0_r + hy_r * i_r_y;
//                            for (std::size_t i_r_x = 0; i_r_x < NxR; ++i_r_x) {
//                                double r_x = x0_r + hx_r * i_r_x;
//                                std::size_t i_r = i_r_y * NxR + i_r_x;
//
//                                times_to_receivers_[i_p * n_receivers_ + i_r] =
//                                        euqlidean_dist(r_x, p_x, r_y, p_y, 0, p_z) / velocity;
//                            }
//                    }


                }
            }
        }
    }

    template <typename T1>
    std::pair<std::unique_ptr<T2[]>, std::unique_ptr<T1[]>>
    generate_user_data_by_receivers(std::vector<double> receivers_coords,
                                    bool is_transpose_receivers) {
	    std::size_t n_receivers = receivers_coords.size();
	    std::unique_ptr<T2[]> times_to_receivers(new T2[z_dim_*y_dim_*x_dim_*n_receivers]);
	    std::unique_ptr<T1[]> gather(new T1[n_receivers * n_samples_]);

	    double z0 = grid_[0].first, z1 = grid_[0].second;
	    double y0 = grid_[1].first, y1 = grid_[1].second;
	    double x0 = grid_[2].first, x1 = grid_[2].second;
	    double dz = (z1 - z0) / (z_dim_ - 1);
	    double dy = (y1 - y0) / (y_dim_ - 1);
	    double dx = (x1 - x0) / (x_dim_ - 1);

	    if (is_transpose_receivers) {
	        for (std::size_t i_r = 0; i_r < n_receivers; ++i_r) {
	            double r_y = receivers_coords[i_r*2], r_x = receivers_coords[i_r*2 + 1];
	            for (std::size_t i_z = 0; i_z < z_dim_; ++i_z) {
	                double p_z = z0 + dz*i_z;
	                for (std::size_t i_y = 0; i_y < y_dim_; ++i_y) {
	                    double p_y = y0 + dy*i_y;
	                    for (std::size_t i_x = 0; i_x < x_dim_; ++i_x) {
                            double p_x = x0 + dx*i_x;

                            std::size_t i_p = i_z*x_dim_ + i_x;

                            times_to_receivers[i_r*z_dim_*y_dim_*x_dim_ + i_p] =
                                euqlidean_dist(r_x, p_x, r_y, p_y, 0, p_z) / velocity_;
                        }
	                }
	            }
	        }
	    } else {

	        for (std::size_t i_z = 0; i_z < z_dim_; ++i_z) {
	            double p_z = z0 + dz*i_z;
	            for (std::size_t i_y = 0; i_y < y_dim_; ++i_y) {
	                double p_y = y0 + dy*i_y;
	                for (std::size_t i_x = 0; i_x < x_dim_; ++i_x) {
	                    double p_x = x0 + dx*i_x;

	                    std::size_t i_p = i_z*x_dim_ + i_x;

	                    for (std::size_t i_r = 0; i_r < n_receivers; ++i_r) {
	                        double r_y = receivers_coords[i_r*2], r_x = receivers_coords[i_r*2 + 1];
	                        times_to_receivers[i_p*n_receivers + i_r] =
	                            euqlidean_dist(r_x, p_x, r_y, p_y, 0, p_z) / velocity_;
	                    }
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

    std::size_t get_y_dim() const {
        return y_dim_;
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

	std::size_t n_receivers_;
	std::size_t x_dim_;
	std::size_t y_dim_;
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

#endif //_TEST_DATA_GENERATE3D_H