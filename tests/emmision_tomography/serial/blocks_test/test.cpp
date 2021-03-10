#include "emission_tomography_method_without_blocks.h"
#include "emission_tomography_method_with_blocks.h"
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

	perf_wrapper(std::bind(coh_sum, std::ref(gather), std::ref(receivers_coords), std::ref(sources_coords), std::ref(sources_receivers_times), dt, tensor_matrix, result_data), measurements_file);

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
	std::size_t n_samples = 2000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	measurements_file << "without blocks;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	run_program(emissionTomographyMethodWithoutBlocks<double>, data_gen, measurements_file);
	measurements_file << std::endl;

	for (std::size_t receivers_block_size = 20; receivers_block_size < NxR*NyR; receivers_block_size += 20) {
		for (std::size_t samples_block_size = 100; samples_block_size < n_samples; samples_block_size += 100) {
			measurements_file << "with blocks;";
			measurements_file << data_gen.get_n_sources() << ";";
			measurements_file << data_gen.get_n_receivers() << ";";
			measurements_file << data_gen.get_n_samples() << ";";
			measurements_file << receivers_block_size << ";";
			measurements_file << samples_block_size << ";";

			std::cerr << "AAAAAAAAAAAAAAA" << std::endl;

			run_program(std::bind(
				emissionTomographyMethodWithBlocks<double>,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3,
				std::placeholders::_4,
				std::placeholders::_5,
				std::placeholders::_6,
				std::placeholders::_7,
				receivers_block_size,
				samples_block_size), data_gen, measurements_file
			);
			measurements_file << std::endl;
		}
	}

}

void test_n_smpls_greater_n_sou(std::ofstream &measurements_file) {
	double x0_r = 100, x1_r = 2000;
	std::size_t NxR = 20;
	double y0_r = 100, y1_r = 2000;
	std::size_t NyR = 20;
	double x0_s = 0, x1_s = 3000;
	std::size_t NxS = 15;
	double y0_s = 0, y1_s = 3000; 
	std::size_t NyS = 15;
	double z0_s = 0, z1_s = 3000;
	std::size_t NzS = 15;
	std::size_t n_samples = 80000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	measurements_file << "without blocks;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	run_program(emissionTomographyMethodWithoutBlocks<double>, data_gen, measurements_file);
	measurements_file << std::endl;

	for (std::size_t receivers_block_size = 20; receivers_block_size < NxR*NyR; receivers_block_size += 20) {
		for (std::size_t samples_block_size = 2000; samples_block_size < n_samples; samples_block_size += 2000) {
			measurements_file << "with blocks;";
			measurements_file << data_gen.get_n_sources() << ";";
			measurements_file << data_gen.get_n_receivers() << ";";
			measurements_file << data_gen.get_n_samples() << ";";
			measurements_file << receivers_block_size << ";";
			measurements_file << samples_block_size << ";";

			run_program(std::bind(
				emissionTomographyMethodWithBlocks<double>,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3,
				std::placeholders::_4,
				std::placeholders::_5,
				std::placeholders::_6,
				std::placeholders::_7,
				receivers_block_size,
				samples_block_size), data_gen, measurements_file
			);
			measurements_file << std::endl;
		}
	}
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
	std::size_t n_samples = 27000;
	double velocity = 3000;

	test_data_generator data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								n_samples, velocity);

	measurements_file << "without blocks;";
	measurements_file << data_gen.get_n_sources() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	run_program(emissionTomographyMethodWithoutBlocks<double>, data_gen, measurements_file);
	measurements_file << std::endl;

	for (std::size_t receivers_block_size = 20; receivers_block_size < NxR*NyR; receivers_block_size += 20) {
		for (std::size_t samples_block_size = 2001; samples_block_size < n_samples; samples_block_size += 2000) {
			measurements_file << "with blocks;";
			measurements_file << data_gen.get_n_sources() << ";";
			measurements_file << data_gen.get_n_receivers() << ";";
			measurements_file << data_gen.get_n_samples() << ";";
			measurements_file << receivers_block_size << ";";
			measurements_file << samples_block_size << ";";

			run_program(std::bind(
				emissionTomographyMethodWithBlocks<double>,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3,
				std::placeholders::_4,
				std::placeholders::_5,
				std::placeholders::_6,
				std::placeholders::_7,
				receivers_block_size,
				samples_block_size), data_gen, measurements_file
			);
			measurements_file << std::endl;
		}
	}
}

void create_measurements_file(const std::string &filename, std::ofstream& measurements_file) {
	measurements_file.open(filename);

	measurements_file << "summation version;";
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

}

int main(int argc, char const *argv[]) {

	std::ofstream measurements_file1;
	std::ofstream measurements_file2;
	std::ofstream measurements_file3;
	// create_measurements_file("../measurements1.csv", measurements_file1);
	create_measurements_file("../measurements2.csv", measurements_file2);
	// create_measurements_file("../measurements3.csv", measurements_file3);

	// test_n_sou_greater_n_smpls(measurements_file1);
	test_n_smpls_greater_n_sou(measurements_file2);
	// test_n_sou_equal_n_smpls(measurements_file3);

	return 0;
}