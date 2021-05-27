#include <iostream>

#include "array2D.h"
#include "emission_tomography_method_cycles_reverses.h"
#include "test_data_generator.h"

void test() {
    double x0_r = 100, x1_r = 2000;
	std::size_t NxR = 20;
	double y0_r = 100, y1_r = 2000;
	std::size_t NyR = 20;
	double x0_s = 0, x1_s = 3000;
	std::size_t NxS = 35;
	double y0_s = 0, y1_s = 3000;
	std::size_t NyS = 35;
	double z0_s = 0, z1_s = 3000;
	std::size_t NzS = 35;
	std::size_t n_samples = 43000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	Array2D<double> gather(data_gen.get_gather(), data_gen.get_n_receivers(), data_gen.get_n_samples());
	Array2D<double> receivers_coords(data_gen.get_receivers_coords(), data_gen.get_n_receivers(), 3);
	Array2D<double> sources_coords(data_gen.get_sources_coords(), data_gen.get_n_sources(), 3);
	Array2D<double> sources_receivers_times(data_gen.get_sources_receivers_times(), data_gen.get_n_sources(), data_gen.get_n_receivers());
	double dt = data_gen.get_dt();
	double tensor_matrix[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

	double *result_data = new double[data_gen.get_n_sources()*data_gen.get_n_samples()];

//	std::vector<std::size_t> times(5);
	emissionTomographyMethodCyclesReverses(gather, receivers_coords, sources_coords, sources_receivers_times, dt, tensor_matrix, result_data);
	std::cerr << result_data[0] << std::endl;

	delete [] result_data;
}

int main(int argc, char *argv[]) {
    test();
    return 0;
}