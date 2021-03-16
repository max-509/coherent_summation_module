#include "arrival_times_by_horizontally_layered_medium3D.h"
#include "kirchhoff_migration_reverse_cycles.h"
#include "kirchhoff_migration_blocks_receivers_inner_loop.h"
#include "kirchhoff_migration_blocks_points_inner_loop.h"
#include "perf_wrapper.h"
#include "test_data_generator3D.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>

template <typename T1, typename T2, typename CalculatorType, typename T>
using CohSumType = std::function<void (const Array2D<T1> &, 
                                const Array2D<T2> &,
                                double, double,
                                double,
                                ArrivalTimesCalculator3D<CalculatorType, T> &,
                                T1 *)>;

void run_program(CohSumType<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double> coh_sum, test_data_generator3D &data_gen, std::ofstream &measurements_file) {
	Array2D<double> gather(data_gen.get_gather(), data_gen.get_n_receivers(), data_gen.get_n_samples());
	Array2D<double> receivers_coords(data_gen.get_receivers_coords(), data_gen.get_n_receivers(), 2);
	double s_x = data_gen.get_s_x(), s_y = data_gen.get_s_y();
	double dt = data_gen.get_dt();

	Array3D<double> velocity_model(data_gen.get_velocity_model(), data_gen.get_z_dim(), data_gen.get_y_dim(), data_gen.get_x_dim());
	ArrivalTimesCalculator3D<ArrivalTimesByHorizontallyLayeredMedium3D<double>, double> arrival_times_calculator(velocity_model, data_gen.get_grid());

	double *result_data = new double[data_gen.get_z_dim()*data_gen.get_y_dim()*data_gen.get_x_dim()];

	perf_wrapper(std::bind(coh_sum, std::ref(gather), std::ref(receivers_coords), s_x, s_y, dt, std::ref(arrival_times_calculator), result_data), measurements_file);

	delete [] result_data;

}

void test_n_sou_greater_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 27;
	double y0_r = 0, y1_r = 4000;
	std::size_t NyR = 27;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 315;
	double y0_s = 0, y1_s = 4000;
	std::size_t NyS = 315;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 315;
	std::size_t n_samples = 20000;
	double s_x = 0.0, s_y = 0.0;
	double dt = 0.0001;
	std::vector<double> velocities = {2000., 3000., 4000., 5000.};
	std::vector<double> borders = {500., 500., 500., 500.};

	test_data_generator3D data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								s_x, s_y, dt,
								n_samples, 
								velocities,
								borders);

	measurements_file << "reverse_cycles;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	run_program(kirchhoffMigrationCHG3DReverseCycles<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	for (std::size_t receivers_block_size = 30; receivers_block_size < data_gen.get_n_receivers(); receivers_block_size += 30) {
		for (std::size_t z_block_size = 35; z_block_size < data_gen.get_z_dim(); z_block_size += 35) {
			for (std::size_t y_block_size = 35; y_block_size < data_gen.get_y_dim(); y_block_size += 35) {
				for (std::size_t x_block_size = 35; x_block_size < data_gen.get_x_dim(); z_block_size += 35) {
					measurements_file << "blocks receivers inner loop;";
					measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
					measurements_file << data_gen.get_n_receivers() << ";";
					measurements_file << data_gen.get_n_samples() << ";";
					measurements_file << receivers_block_size << ";";
					measurements_file << z_block_size << ";";
					measurements_file << y_block_size << ";";
					measurements_file << x_block_size << ";";

					run_program(std::bind(
						kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>,
						std::placeholders::_1,
						std::placeholders::_2,
						std::placeholders::_3,
						std::placeholders::_4,
						std::placeholders::_5,
						std::placeholders::_6,
						std::placeholders::_7,
						receivers_block_size,
						z_block_size,
						y_block_size,
						x_block_size), data_gen, measurements_file
					);
					measurements_file << std::endl;

					measurements_file << "blocks points inner loop;";
					measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
					measurements_file << data_gen.get_n_receivers() << ";";
					measurements_file << data_gen.get_n_samples() << ";";
					measurements_file << receivers_block_size << ";";
					measurements_file << z_block_size << ";";
					measurements_file << y_block_size << ";";
					measurements_file << x_block_size << ";";

					run_program(std::bind(
						kirchhoffMigrationCHG3DBlocksPointsInnerLoop<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>,
						std::placeholders::_1,
						std::placeholders::_2,
						std::placeholders::_3,
						std::placeholders::_4,
						std::placeholders::_5,
						std::placeholders::_6,
						std::placeholders::_7,
						receivers_block_size,
						z_block_size,
						y_block_size,
						x_block_size), data_gen, measurements_file
					);
					measurements_file << std::endl;
				}
			}
		}
	}

}

void test_n_smpls_greater_n_sou(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 25;
	double y0_r = 0, y1_r = 4000;
	std::size_t NyR = 25;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 120;
	double y0_s = 0, y1_s = 4000; 
	std::size_t NyS = 120;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 120;
	std::size_t n_samples = 2000000;
	double s_x = 0.0, s_y = 0.0;
	double dt = 0.000001;
	std::vector<double> velocities = {2000., 3000., 4000., 5000.};
	std::vector<double> borders = {500., 500., 500., 500.};

	test_data_generator3D data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								s_x, s_y, dt,
								n_samples, 
								velocities,
								borders);

	measurements_file << "reverse_cycles;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	run_program(kirchhoffMigrationCHG3DReverseCycles<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	for (std::size_t receivers_block_size = 30; receivers_block_size < data_gen.get_n_receivers(); receivers_block_size += 30) {
		for (std::size_t z_block_size = 16; z_block_size < data_gen.get_z_dim(); z_block_size += 16) {
			for (std::size_t y_block_size = 16; y_block_size < data_gen.get_y_dim(); y_block_size += 16) {
				for (std::size_t x_block_size = 16; x_block_size < data_gen.get_x_dim(); z_block_size += 16) {
					measurements_file << "blocks receivers inner loop;";
					measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
					measurements_file << data_gen.get_n_receivers() << ";";
					measurements_file << data_gen.get_n_samples() << ";";
					measurements_file << receivers_block_size << ";";
					measurements_file << z_block_size << ";";
					measurements_file << y_block_size << ";";
					measurements_file << x_block_size << ";";

					run_program(std::bind(
						kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>,
						std::placeholders::_1,
						std::placeholders::_2,
						std::placeholders::_3,
						std::placeholders::_4,
						std::placeholders::_5,
						std::placeholders::_6,
						std::placeholders::_7,
						receivers_block_size,
						z_block_size,
						y_block_size,
						x_block_size), data_gen, measurements_file
					);
					measurements_file << std::endl;

					measurements_file << "blocks points inner loop;";
					measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
					measurements_file << data_gen.get_n_receivers() << ";";
					measurements_file << data_gen.get_n_samples() << ";";
					measurements_file << receivers_block_size << ";";
					measurements_file << z_block_size << ";";
					measurements_file << y_block_size << ";";
					measurements_file << x_block_size << ";";

					run_program(std::bind(
						kirchhoffMigrationCHG3DBlocksPointsInnerLoop<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>,
						std::placeholders::_1,
						std::placeholders::_2,
						std::placeholders::_3,
						std::placeholders::_4,
						std::placeholders::_5,
						std::placeholders::_6,
						std::placeholders::_7,
						receivers_block_size,
						z_block_size,
						y_block_size,
						x_block_size), data_gen, measurements_file
					);
					measurements_file << std::endl;
				}
			}
		}
	}
}

void test_n_sou_equal_n_smpls(std::ofstream &measurements_file) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 15;
	double y0_r = 0, y1_r = 4000;
	std::size_t NyR = 15;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 185;
	double y0_s = 0, y1_s = 4000; 
	std::size_t NyS = 185;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 185;
	std::size_t n_samples = 6250000;
	double s_x = 0.0, s_y = 0.0;
	double dt = 0.00000032;
	std::vector<double> velocities = {2000., 3000., 4000., 5000.};
	std::vector<double> borders = {500., 500., 500., 500.};

	test_data_generator3D data_gen(x0_r, x1_r, NxR,
								y0_r, y1_r, NyR,
								x0_s, x1_s, NxS,
								y0_s, y1_s, NyS,
								z0_s, z1_s, NzS,
								s_x, s_y, dt,
								n_samples, 
								velocities,
								borders);

	measurements_file << "reverse_cycles;";
	measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
	measurements_file << data_gen.get_n_receivers() << ";";
	measurements_file << data_gen.get_n_samples() << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	measurements_file << 0 << ";";
	run_program(kirchhoffMigrationCHG3DReverseCycles<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>, data_gen, measurements_file);
	measurements_file << std::endl;

	for (std::size_t receivers_block_size = 20; receivers_block_size < data_gen.get_n_receivers(); receivers_block_size += 20) {
		for (std::size_t z_block_size = 20; z_block_size < data_gen.get_z_dim(); z_block_size += 20) {
			for (std::size_t y_block_size = 20; y_block_size < data_gen.get_y_dim(); y_block_size += 20) {
				for (std::size_t x_block_size = 20; x_block_size < data_gen.get_x_dim(); z_block_size += 20) {
					measurements_file << "blocks receivers inner loop;";
					measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
					measurements_file << data_gen.get_n_receivers() << ";";
					measurements_file << data_gen.get_n_samples() << ";";
					measurements_file << receivers_block_size << ";";
					measurements_file << z_block_size << ";";
					measurements_file << y_block_size << ";";
					measurements_file << x_block_size << ";";

					run_program(std::bind(
						kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>,
						std::placeholders::_1,
						std::placeholders::_2,
						std::placeholders::_3,
						std::placeholders::_4,
						std::placeholders::_5,
						std::placeholders::_6,
						std::placeholders::_7,
						receivers_block_size,
						z_block_size,
						y_block_size,
						x_block_size), data_gen, measurements_file
					);
					measurements_file << std::endl;

					measurements_file << "blocks points inner loop;";
					measurements_file << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
					measurements_file << data_gen.get_n_receivers() << ";";
					measurements_file << data_gen.get_n_samples() << ";";
					measurements_file << receivers_block_size << ";";
					measurements_file << z_block_size << ";";
					measurements_file << y_block_size << ";";
					measurements_file << x_block_size << ";";

					run_program(std::bind(
						kirchhoffMigrationCHG3DBlocksPointsInnerLoop<double, double, ArrivalTimesByHorizontallyLayeredMedium3D<double>, double>,
						std::placeholders::_1,
						std::placeholders::_2,
						std::placeholders::_3,
						std::placeholders::_4,
						std::placeholders::_5,
						std::placeholders::_6,
						std::placeholders::_7,
						receivers_block_size,
						z_block_size,
						y_block_size,
						x_block_size), data_gen, measurements_file
					);
					measurements_file << std::endl;
				}
			}
		}
	}
}

void create_measurements_file(const std::string &filename, std::ofstream& measurements_file) {
	measurements_file.open(filename);

	measurements_file << "summation version;";
	measurements_file << "number of points;";
	measurements_file << "number of receivers;";
	measurements_file << "number of samples;";
	measurements_file << "receivers block size;";
	measurements_file << "z block size;";
	measurements_file << "y block size;";
	measurements_file << "x block size;";
	for (std::size_t i_e = 0; i_e < Events::COUNT_EVENTS; ++i_e) {
		measurements_file << Events::events_names[i_e] << ";";
	}
	measurements_file << "time, s";
	measurements_file << std::endl;

}

int main(int argc, char const *argv[]) {

	std::ofstream measurements_file_sou_greater_samples;
	std::ofstream measurements_file_samples_greater_sou;
	std::ofstream measurements_file_sou_equal_samples;

	create_measurements_file("../measurements3D_sources_greater_samples.csv", measurements_file_sou_greater_samples);
	create_measurements_file("../measurements3D_samples_greater_sources.csv", measurements_file_samples_greater_sou);
	create_measurements_file("../measurements3D_sources_equals_samples.csv", measurements_file_sou_equal_samples);

	test_n_sou_greater_n_smpls(measurements_file_sou_greater_samples);
	test_n_smpls_greater_n_sou(measurements_file_samples_greater_sou);
	test_n_sou_equal_n_smpls(measurements_file_sou_equal_samples);

	return 0;
}