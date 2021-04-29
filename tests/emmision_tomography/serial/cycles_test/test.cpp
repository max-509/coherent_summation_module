#include "emission_tomography_method_native.h"
#include "emission_tomography_method_cycles_reverses.h"
#include "perf_wrapper.h"
#include "test_data_generator.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>

template <typename T>
using CohSumType = std::function<void (const Array2D<T> &, 
                                const Array2D<T> &,
                                const Array2D<T> &,
                                const Array2D<T> &,
                                double,
                                const T *,
                                T *)>;

void run_program(CohSumType<double> coh_sum, test_data_generator &data_gen, std::ofstream &measurements_file) {
	Array2D<double> gather(data_gen.get_gather(), data_gen.get_n_receivers(), data_gen.get_n_samples());
	Array2D<double> receivers_coords(data_gen.get_receivers_coords(), data_gen.get_n_receivers(), 3);
	Array2D<double> sources_coords(data_gen.get_sources_coords(), data_gen.get_n_sources(), 3);
	Array2D<double> sources_receivers_times(data_gen.get_sources_receivers_times(), data_gen.get_n_sources(), data_gen.get_n_receivers());
	double dt = data_gen.get_dt();
	double tensor_matrix[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

	double *result_data = new double[data_gen.get_n_sources()*data_gen.get_n_samples()];

	auto res = perf_wrapper(std::bind(coh_sum, std::ref(gather), std::ref(receivers_coords), std::ref(sources_coords), std::ref(sources_receivers_times), dt, tensor_matrix, result_data));

	for (const auto &e : res.second) {
	    measurements_file << e << ";";
	}
	measurements_file << res.first;

	delete [] result_data;

}

void test_n_sou_greater_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 100, x1_r = 2000;
	std::size_t NxR = 20;
	double y0_r = 100, y1_r = 2000;
	std::size_t NyR = 20;
	double x0_s = 0, x1_s = 3000;
	std::size_t NxS = 50;
	double y0_s = 0, y1_s = 3000; 
	std::size_t NyS = 50;
	double z0_s = 0, z1_s = 3000;
	std::size_t NzS = 50;
	std::size_t n_samples = 4000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	measurements_file << "native_vect;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(emissionTomographyMethodNative<double, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	measurements_file << "reverses;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(emissionTomographyMethodCyclesReverses<double, double>, data_gen, measurements_file);
	measurements_file << std::endl;

}

void test_n_smpls_greater_n_sou(std::ofstream &measurements_file) {
	double x0_r = 100, x1_r = 2000;
	std::size_t NxR = 20;
	double y0_r = 100, y1_r = 2000;
	std::size_t NyR = 20;
	double x0_s = 0, x1_s = 3000;
	std::size_t NxS = 20;
	double y0_s = 0, y1_s = 3000; 
	std::size_t NyS = 20;
	double z0_s = 0, z1_s = 3000;
	std::size_t NzS = 20;
	std::size_t n_samples = 60000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	measurements_file << "native_vect;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(emissionTomographyMethodNative<double, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	measurements_file << "reverses;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(emissionTomographyMethodCyclesReverses<double, double>, data_gen, measurements_file);
	measurements_file << std::endl;
}

void test_n_sou_equal_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 100, x1_r = 2000;
	std::size_t NxR = 20;
	double y0_r = 100, y1_r = 2000;
	std::size_t NyR = 20;
	double x0_s = 0, x1_s = 3000;
	std::size_t NxS = 30;
	double y0_s = 0, y1_s = 3000; 
	std::size_t NyS = 30;
	double z0_s = 0, z1_s = 3000;
	std::size_t NzS = 30;
	std::size_t n_samples = 20000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	measurements_file << "native_vect;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(emissionTomographyMethodNative<double, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	measurements_file << "reverses;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(emissionTomographyMethodCyclesReverses<double, double>, data_gen, measurements_file);
	measurements_file << std::endl;
}

int main(int argc, char const *argv[]) {

	std::ofstream measurements_file("./measurements.csv");
	measurements_file << "summation version;";
	measurements_file << "number of sources;";
	measurements_file << "number of receivers;";
	measurements_file << "number of samples;";
	for (std::size_t i_e = 0; i_e < Events::COUNT_EVENTS; ++i_e) {
		measurements_file << Events::events_names[i_e] << ";";
	}
	measurements_file << "time, s";
	measurements_file << std::endl;

	test_n_sou_greater_n_smpls(measurements_file);
	test_n_smpls_greater_n_sou(measurements_file);
	test_n_sou_equal_n_smpls(measurements_file);

	return 0;
}