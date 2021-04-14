#include "kirchhoff_migration_auto_vectorization.h"
#include "kirchhoff_migration_manual_vectorization.h"
#include "perf_wrapper.h"
#include "test_data_generator2D.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>

#ifdef __AVX512F__
#define SIMD_EXTENSION "AVX512F"
#elif __AVX2__
#define SIMD_EXTENSION "AVX2"
#elif __SSE4_2__
#define SIMD_EXTENSION "SSE4_2"
#else
#define SIMD_EXTENSION "NO_SIMD_EXTENSIONS"
#endif

template <typename T1, typename T2>
using CohSumType = std::function<void (const Array2D<T1> &, 
                                const std::vector<T2> &,
                                const Array2D<T2> &,
                                std::ptrdiff_t, std::ptrdiff_t,
                                double,
                                T1 *)>;

void run_program(const CohSumType<double, double>& coh_sum,
                 test_data_generator2D<double> &data_gen,
                 std::ofstream &measurements_file,
                 double x0_r, double x1_r, std::size_t NxR,
                 std::size_t receivers_step,
                 bool is_trans) {

    auto times_to_source = data_gen.get_times_to_source();
	std::ptrdiff_t z_dim = data_gen.get_z_dim(), x_dim = data_gen.get_x_dim();
	double dt = data_gen.get_dt();

	double *result_data = new double[z_dim*x_dim];

	double dx_r = (x1_r - x0_r) / (NxR - 1);

	std::vector<double> receivers_coords(receivers_step);

	std::vector<uint64_t> events_counts(Events::COUNT_EVENTS, 0);

	double time_in_sec = 0.0;

	for (std::size_t rec_bl = 0; rec_bl < NxR; rec_bl += receivers_step) {
	    for (std::size_t i_r = rec_bl; i_r < std::min(NxR, rec_bl + receivers_step); ++i_r) {
	        receivers_coords[i_r - rec_bl] = x0_r + i_r*dx_r;
	    }

	    auto user_datas = data_gen.generate_user_data_by_receivers<double>(receivers_coords, is_trans);

	    Array2D<double> gather(user_datas.second, receivers_step, data_gen.get_n_samples());

	    std::pair<double, std::vector<uint64_t>> res;
	    if (is_trans) {
	        Array2D<double> times_to_receivers(user_datas.first.get(), receivers_step, z_dim*x_dim);
	        res = perf_wrapper(std::bind(coh_sum, std::ref(gather), std::ref(times_to_source), std::ref(times_to_receivers), z_dim, x_dim, dt, result_data));
	    } else {
	        Array2D<double> times_to_receivers(user_datas.first.get(), z_dim*x_dim, receivers_step);
	        res = perf_wrapper(std::bind(coh_sum, std::ref(gather), std::ref(times_to_source), std::ref(times_to_receivers), z_dim, x_dim, dt, result_data));
	    }

	    for (std::size_t i = 0; i < events_counts.size(); ++i) {
	        events_counts[i] += res.second[i];
	    }

	    time_in_sec += res.first;
	}

	for (const auto &e : events_counts) {
	    measurements_file << e << ";";
	}
	measurements_file << time_in_sec;

	delete [] result_data;

}

void test_n_sou_greater_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 1800;
	std::size_t receivers_step = 20;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 6500;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 6500;
	std::size_t n_samples = 40000;
	double velocity = 3500.0;
	double s_x = 0.0;
	double dt = 0.002;


	test_data_generator2D<double> data_gen(x0_s, x1_s, NxS,
                                            z0_s, z1_s, NzS,
                                            s_x, dt,
                                            n_samples,
                                            velocity);

	measurements_file << "auto vect;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << NxR << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DAutoVectorization<double, double>,
	        data_gen,
	        measurements_file,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
	measurements_file << std::endl;

	measurements_file << "manual vect;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << NxR << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DManualVectorization<double, double>,
	        data_gen,
	        measurements_file,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
	measurements_file << std::endl;

}

void test_n_smpls_greater_n_sou(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 3000;
	std::size_t receivers_step = 50;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 2500;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 2500;
	std::size_t n_samples = 10000000;
	double velocity = 3500.0;
	double s_x = 0.0;
	double dt = 0.002;

	test_data_generator2D<double> data_gen(x0_s, x1_s, NxS,
                                            z0_s, z1_s, NzS,
                                            s_x, dt,
                                            n_samples,
                                            velocity);

	measurements_file << "auto vect;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << NxR << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DAutoVectorization<double, double>,
	        data_gen,
	        measurements_file,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
	measurements_file << std::endl;

	measurements_file << "manual vect;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << NxR << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DManualVectorization<double, double>,
	        data_gen,
	        measurements_file,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
	measurements_file << std::endl;
}

void test_n_sou_equal_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 2200;
	std::size_t receivers_step = 50;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 4000;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 4000;
	std::size_t n_samples = 16000000;
	double velocity = 3500.0;
	double s_x = 0.0;
	double dt = 0.002;

	test_data_generator2D<double> data_gen(x0_s, x1_s, NxS,
                                            z0_s, z1_s, NzS,
                                            s_x, dt,
                                            n_samples,
                                            velocity);

	measurements_file << "auto vect;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << NxR << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DAutoVectorization<double, double>,
	        data_gen,
	        measurements_file,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
	measurements_file << std::endl;

	measurements_file << "manual vect;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << NxR << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DManualVectorization<double, double>,
	        data_gen,
	        measurements_file,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
	measurements_file << std::endl;
}

int main(int argc, char const *argv[]) {

	std::ofstream measurements_file(std::string("./measurements2D_") + SIMD_EXTENSION + ".csv");
	measurements_file << "summation version;";
	measurements_file << "number of points;";
	measurements_file << "number of receivers;";
	measurements_file << "number of samples;";
	for (std::size_t i_e = 0; i_e < Events::COUNT_EVENTS; ++i_e) {
		measurements_file << Events::events_names[i_e] << ";";
	}
	measurements_file << "time, s";
	measurements_file << std::endl;

	test_n_sou_greater_n_smpls(measurements_file);
//	test_n_smpls_greater_n_sou(measurements_file);
//	test_n_sou_equal_n_smpls(measurements_file);

	return 0;
}