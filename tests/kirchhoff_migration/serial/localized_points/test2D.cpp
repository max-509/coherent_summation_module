#include "kirchhoff_migration_native.h"
#include "arrival_times_by_horizontally_layered_medium2D.h"
#include "arrival_times_calculator2D.h"
#include "perf_wrapper.h"
#include "test_data_generator2D.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>

template <typename T1, typename T2, typename CalculatorType, typename T>
using CohSumType = std::function<void (const Array2D<T1> &, 
                                const std::vector<T2> &,
                                double,
                                double,
                                ArrivalTimesCalculator2D<CalculatorType, T> &,
                                T1 *)>;

void run_program(CohSumType<double, double, ArrivalTimesByHorizontallyLayeredMedium2D<double>, double> coh_sum, test_data_generator2D &data_gen, std::ofstream &measurements_file) {
	Array2D<double> gather(data_gen.get_gather(), data_gen.get_n_receivers(), data_gen.get_n_samples());
	std::vector<double> &receivers_coords(data_gen.get_receivers_coords());
	double s_x = data_gen.get_s_x();
	double dt = data_gen.get_dt();

	Array2D<double> velocity_model(data_gen.get_velocity_model(), data_gen.get_z_dim(), data_gen.get_x_dim());
	ArrivalTimesCalculator2D<ArrivalTimesByHorizontallyLayeredMedium2D<double>, double> arrival_times_calculator(velocity_model, data_gen.get_grid());

	double *result_data = new double[data_gen.get_z_dim()*data_gen.get_x_dim()];

	perf_wrapper(std::bind(coh_sum, std::ref(gather), std::ref(receivers_coords), s_x, dt, std::ref(arrival_times_calculator), result_data), measurements_file);

	delete [] result_data;

}

void test_n_sou_greater_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 400;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 500;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 250;
	std::size_t n_samples = 8000;
	double s_x = 0.0;
	double dt = 0.001;
	std::vector<double> velocities = {2000., 3000., 4000., 5000.};
	std::vector<double> borders = {500., 500., 500., 500.};

	test_data_generator2D data_gen(x0_r, x1_r, NxR,
								x0_s, x1_s, NxS,
								z0_s, z1_s, NzS,
								s_x, dt,
								n_samples, 
								velocities,
								borders);

	measurements_file << "native;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DNative<double, double, ArrivalTimesByHorizontallyLayeredMedium2D<double>, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	//TODO: 
	// measurements_file << "localized points;";
	// measurements_file << data_gen.get_x_dim()*get_z_dim() << ";";
	// measurements_file << data_gen.get_n_receivers() << ";";
	// measurements_file << data_gen.get_n_samples() << ";";
	// run_program(emissionTomographyMethodCyclesReverses<double>, data_gen, measurements_file);
	// measurements_file << std::endl;

}

void test_n_smpls_greater_n_sou(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 400;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 90;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 90;
	std::size_t n_samples = 100000;
	double s_x = 0.0;
	double dt = 0.001;
	std::vector<double> velocities = {2000., 3000., 4000., 5000.};
	std::vector<double> borders = {500., 500., 500., 500.};

	test_data_generator2D data_gen(x0_r, x1_r, NxR,
								x0_s, x1_s, NxS,
								z0_s, z1_s, NzS,
								s_x, dt,
								n_samples, 
								velocities,
								borders);

	measurements_file << "native;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DNative<double, double, ArrivalTimesByHorizontallyLayeredMedium2D<double>, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	//TODO:
	// measurements_file << "reverses;";
	// measurements_file << data_gen.get_n_sources() << ";";
	// measurements_file << data_gen.get_n_receivers() << ";";
	// measurements_file << data_gen.get_n_samples() << ";";
	// run_program(emissionTomographyMethodCyclesReverses<double>, data_gen, measurements_file);
	// measurements_file << std::endl;
}

void test_n_sou_equal_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 400;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 200;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 200;
	std::size_t n_samples = 40000;
	double s_x = 0.0;
	double dt = 0.001;
	std::vector<double> velocities = {2000., 3000., 4000., 5000.};
	std::vector<double> borders = {500., 500., 500., 500.};

	test_data_generator2D data_gen(x0_r, x1_r, NxR,
								x0_s, x1_s, NxS,
								z0_s, z1_s, NzS,
								s_x, dt,
								n_samples, 
								velocities,
								borders);

	measurements_file << "native;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	run_program(kirchhoffMigrationCHG2DNative<double, double, ArrivalTimesByHorizontallyLayeredMedium2D<double>, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	//TODO: 
	// measurements_file << "reverses;";
	// measurements_file << data_gen.get_n_sources() << ";";
	// measurements_file << data_gen.get_n_receivers() << ";";
	// measurements_file << data_gen.get_n_samples() << ";";
	// run_program(emissionTomographyMethodCyclesReverses<double>, data_gen, measurements_file);
	// measurements_file << std::endl;
}

int main(int argc, char const *argv[]) {

	std::ofstream measurements_file("../measurements2D.csv");
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
	test_n_smpls_greater_n_sou(measurements_file);
	test_n_sou_equal_n_smpls(measurements_file);

	return 0;
}