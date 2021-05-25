#include "emission_tomography_method_without_vectorization.h"
#include "perf_wrapper.h"
#include "test_data_generator.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <omp.h>

#ifdef __AVX512F__
#define SIMD_EXTENSION "AVX512F"
#elif __AVX2__
#define SIMD_EXTENSION "AVX2"
#elif __SSE2__
#define SIMD_EXTENSION "SSE2"
#else
#define SIMD_EXTENSION "NO_SIMD_EXTENSIONS"
#endif

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

inline bool is_exist_file(const std::string &filename) {
	std::ifstream f(filename);
	return f.good();
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
	std::size_t n_samples = 8000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);
	/**/
	std::ptrdiff_t best_receivers_block_size = 190;
	std::ptrdiff_t best_samples_block_size = 1600;
	/**/

	measurements_file << "parallel by sources-samples;";
	measurements_file << "auto;";
	measurements_file << SIMD_EXTENSION << ";";
	measurements_file << omp_get_max_threads() << ";";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << best_receivers_block_size << ";";
	measurements_file << best_samples_block_size << ";";

	run_program(std::bind(
				emissionTomographyMethodWithoutVectorization<double, double>,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3,
				std::placeholders::_4,
				std::placeholders::_5,
				std::placeholders::_6,
				std::placeholders::_7,
				best_receivers_block_size,
				best_samples_block_size), data_gen, measurements_file);
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
	std::size_t n_samples = 100000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	/**/
	std::ptrdiff_t best_receivers_block_size = 20;
	std::ptrdiff_t best_samples_block_size = 11000;
	/**/

	measurements_file << "parallel by sources-samples;";
	measurements_file << "auto;";
	measurements_file << SIMD_EXTENSION << ";";
	measurements_file << omp_get_max_threads() << ";";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << best_receivers_block_size << ";";
	measurements_file << best_samples_block_size << ";";

	run_program(std::bind(
				emissionTomographyMethodWithoutVectorization<double, double>,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3,
				std::placeholders::_4,
				std::placeholders::_5,
				std::placeholders::_6,
				std::placeholders::_7,
				best_receivers_block_size,
				best_samples_block_size), data_gen, measurements_file);
	measurements_file << std::endl;
}

void test_n_sou_equal_n_smpls(std::ofstream &measurements_file) {
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

	/**/
	std::ptrdiff_t best_receivers_block_size = 50;
	std::ptrdiff_t best_samples_block_size = 8000;
	/**/

	measurements_file << "parallel by sources-samples;";
	measurements_file << "auto;";
	measurements_file << SIMD_EXTENSION << ";";
	measurements_file << omp_get_max_threads() << ";";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << best_receivers_block_size << ";";
	measurements_file << best_samples_block_size << ";";
	
	run_program(std::bind(
				emissionTomographyMethodWithoutVectorization<double, double>,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3,
				std::placeholders::_4,
				std::placeholders::_5,
				std::placeholders::_6,
				std::placeholders::_7,
				best_receivers_block_size,
				best_samples_block_size), data_gen, measurements_file);
	measurements_file << std::endl;
}

int main(int argc, char const *argv[]) {

	std::string filename = std::string("./measurements") + SIMD_EXTENSION + ".csv";

	std::ofstream measurements_file;
	if (!is_exist_file(filename)) {
		measurements_file.open(filename);

		measurements_file << "summation version;";
		measurements_file << "type vectorization;";
		measurements_file << "SIMD extension;";
		measurements_file << "number of threads;";
		measurements_file << "number of sources;";
		measurements_file << "number of receivers;";
		measurements_file << "number of samples;";
		measurements_file << "receivers block size;";
		measurements_file << "samples block size;";
		for (std::size_t i_e = 0; i_e < Events::COUNT_EVENTS; ++i_e) {
			measurements_file << Events::events_names[i_e] << ";";
		}
		measurements_file << "time, s";
		measurements_file << std::endl;
	} else {
		measurements_file.open(filename, std::ios::app | std::ios::out);		
	}

	test_n_sou_greater_n_smpls(measurements_file);
	test_n_smpls_greater_n_sou(measurements_file);
	test_n_sou_equal_n_smpls(measurements_file);

	return 0;
}