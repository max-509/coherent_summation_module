#include <iostream>

#include "array2D.h"
#include "kirchhoff_migration_manual_vectorization.h"
#include "test_data_generator3D.h"

void test() {
    double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 46;
	double y0_r = 0, y1_r = 4000;
	std::size_t NyR = 46;
	std::size_t receivers_step = 50;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 250;
	double y0_s = 0, y1_s = 4000;
	std::size_t NyS = 250;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 250;
	std::size_t n_samples = 40000;
	double velocity = 3500.0;
	double s_x = 0.0, s_y = 0.0;
	double dt = 0.0002;

	test_data_generator3D<double> data_gen(x0_s, x1_s, NxS,
								        y0_s, y1_s, NyS,
								        z0_s, z1_s, NzS,
								        s_x, s_y, dt,
								        n_samples,
								        velocity);

	auto times_to_source = data_gen.get_times_to_source();
	std::ptrdiff_t z_dim = data_gen.get_z_dim(), y_dim = data_gen.get_y_dim(), x_dim = data_gen.get_x_dim();

	double *result_data = new double[z_dim*y_dim*x_dim];

	double dx_r = (x1_r - x0_r) / (NxR - 1);
	double dy_r = (y1_r - y0_r) / (NyR - 1);

	std::vector<double> receivers_coords(NxR*NyR*2);

	for (std::size_t i_r_y = 0; i_r_y < NyR; ++i_r_y) {
	    double y_r = y0_r + dy_r*i_r_y;
	    for (std::size_t i_r_x = 0; i_r_x < NxR; ++i_r_x) {
	        double x_r = x0_r + dx_r*i_r_x;
	        receivers_coords[(i_r_y*NxR + i_r_x)*2 + 0] = y_r;
	        receivers_coords[(i_r_y*NxR + i_r_x)*2 + 1] = x_r;
        }
	}

	std::vector<std::pair<double, double>> receivers_coords_block(receivers_step);
    for (std::size_t i_r = 0; i_r < receivers_step; ++i_r) {
        receivers_coords_block[i_r].first = receivers_coords[(i_r) * 2];
        receivers_coords_block[i_r].second = receivers_coords[(i_r) * 2 + 1];
    }

    auto user_datas = data_gen.generate_user_data_by_receivers<double>(receivers_coords_block, true);
    Array2D<double> gather(user_datas.second, receivers_step, data_gen.get_n_samples());
    Array2D<double> times_to_receivers(user_datas.first.get(), z_dim*y_dim*x_dim, receivers_step);

	for (std::ptrdiff_t rec_bl = 0; rec_bl < NxR*NyR; rec_bl += receivers_step) {
	    kirchhoffMigrationCHG3DManualVectorization(gather, times_to_source, times_to_receivers, z_dim, y_dim, x_dim, dt, result_data);
	}
	std::cerr << result_data[0] << std::endl;

	delete [] result_data;
}

int main(int argc, char *argv[]) {
    test();
    return 0;
}
