#include "kirchhoff_migration_reverse_cycles.h"
#include "kirchhoff_migration_blocks_receivers_inner_loop.h"
#include "kirchhoff_migration_blocks_points_inner_loop.h"
#include "perf_wrapper.h"
#include "test_data_generator2D.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>
#include <initializer_list>

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

void test_n_sou_greater_n_smpls(std::ofstream &m_f1, std::ofstream &m_f2, std::ofstream &m_f3) {
	double x0_r = 0, x1_r = 4000;
	std::size_t NxR = 500;
	std::size_t receivers_step = 50;
	double x0_s = 0, x1_s = 4000;
	std::size_t NxS = 6800;
	double z0_s = 0, z1_s = 2000;
	std::size_t NzS = 6800;
	std::size_t n_samples = 20000;
	double velocity = 3500.0;
	double s_x = 0.0;
	double dt = 0.002;

	test_data_generator2D<double> data_gen(x0_s, x1_s, NxS,
                                            z0_s, z1_s, NzS,
                                            s_x, dt,
                                            n_samples,
                                            velocity);

//	m_f1 << "reverse_cycles;";
//    m_f1 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
//    m_f1 << NxR << ";";
//    m_f1 << data_gen.get_n_samples() << ";";
//    m_f1 << 0 << ";";
//	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
//	        data_gen,
//                m_f1,
//	        x0_r, x1_r, NxR,
//	        receivers_step,
//	        true);
//    m_f1 << std::endl;
//
//    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
//        m_f1 << "inner points;";
//        m_f1 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
//        m_f1 << NxR << ";";
//        m_f1 << data_gen.get_n_samples() << ";";
//        m_f1 << p_block_size << ";";
//        run_program(std::bind(kirchhoffMigrationCHG2DBlocksPointsInnerLoop<double, double>,
//                std::placeholders::_1,
//                std::placeholders::_2,
//                std::placeholders::_3,
//                std::placeholders::_4,
//                std::placeholders::_5,
//                std::placeholders::_6,
//                std::placeholders::_7,
//                p_block_size),
//                data_gen,
//                    m_f1,
//                x0_r, x1_r, NxR,
//                receivers_step,
//                true);
//        m_f1 << std::endl;
//    }

//    m_f3 << "reverse_cycles;";
//    m_f3 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
//    m_f3 << NxR << ";";
//    m_f3 << data_gen.get_n_samples() << ";";
//    m_f3 << 0 << ";";
//	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
//	        data_gen,
//                m_f3,
//	        x0_r, x1_r, NxR,
//	        receivers_step,
//	        true);
//    m_f3 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f3 << "strip mining;";
        m_f3 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f3 << NxR << ";";
        m_f3 << data_gen.get_n_samples() << ";";
        m_f3 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksPointsInnerLoopStripMining<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f3,
                x0_r, x1_r, NxR,
                receivers_step,
                true);
        m_f3 << std::endl;
    }

    m_f2 << "reverse_cycles;";
    m_f2 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f2 << NxR << ";";
    m_f2 << data_gen.get_n_samples() << ";";
    m_f2 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f2,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f2 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f2 << "inner receivers;";
        m_f2 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f2 << NxR << ";";
        m_f2 << data_gen.get_n_samples() << ";";
        m_f2 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksReceiversInnerLoop<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f2,
                x0_r, x1_r, NxR,
                receivers_step,
                false);
        m_f2 << std::endl;
    }
}

void test_n_smpls_greater_n_sou(std::ofstream &m_f1, std::ofstream &m_f2, std::ofstream &m_f3) {
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

	m_f1 << "reverse_cycles;";
    m_f1 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f1 << NxR << ";";
    m_f1 << data_gen.get_n_samples() << ";";
    m_f1 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f1,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f1 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f1 << "inner points;";
        m_f1 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f1 << NxR << ";";
        m_f1 << data_gen.get_n_samples() << ";";
        m_f1 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksPointsInnerLoop<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f1,
                x0_r, x1_r, NxR,
                receivers_step,
                true);
        m_f1 << std::endl;
    }

    m_f3 << "reverse_cycles;";
    m_f3 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f3 << NxR << ";";
    m_f3 << data_gen.get_n_samples() << ";";
    m_f3 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f3,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f3 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f3 << "strip mining;";
        m_f3 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f3 << NxR << ";";
        m_f3 << data_gen.get_n_samples() << ";";
        m_f3 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksPointsInnerLoopStripMining<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f3,
                x0_r, x1_r, NxR,
                receivers_step,
                true);
        m_f3 << std::endl;
    }

    m_f2 << "reverse_cycles;";
    m_f2 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f2 << NxR << ";";
    m_f2 << data_gen.get_n_samples() << ";";
    m_f2 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f2,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f2 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f2 << "inner receivers;";
        m_f2 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f2 << NxR << ";";
        m_f2 << data_gen.get_n_samples() << ";";
        m_f2 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksReceiversInnerLoop<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f2,
                x0_r, x1_r, NxR,
                receivers_step,
                false);
        m_f2 << std::endl;
    }


}

void test_n_sou_equal_n_smpls(std::ofstream &m_f1, std::ofstream &m_f2, std::ofstream &m_f3) {
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

    m_f1 << "reverse_cycles;";
    m_f1 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f1 << NxR << ";";
    m_f1 << data_gen.get_n_samples() << ";";
    m_f1 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f1,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f1 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f1 << "inner points;";
        m_f1 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f1 << NxR << ";";
        m_f1 << data_gen.get_n_samples() << ";";
        m_f1 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksPointsInnerLoop<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f1,
                x0_r, x1_r, NxR,
                receivers_step,
                true);
        m_f1 << std::endl;
    }

    m_f3 << "reverse_cycles;";
    m_f3 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f3 << NxR << ";";
    m_f3 << data_gen.get_n_samples() << ";";
    m_f3 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f3,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f3 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f3 << "strip mining;";
        m_f3 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f3 << NxR << ";";
        m_f3 << data_gen.get_n_samples() << ";";
        m_f3 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksPointsInnerLoopStripMining<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f3,
                x0_r, x1_r, NxR,
                receivers_step,
                true);
        m_f3 << std::endl;
    }

    m_f2 << "reverse_cycles;";
    m_f2 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
    m_f2 << NxR << ";";
    m_f2 << data_gen.get_n_samples() << ";";
    m_f2 << 0 << ";";
	run_program(kirchhoffMigrationCHG2DReverseCycles<double, double>,
	        data_gen,
                m_f2,
	        x0_r, x1_r, NxR,
	        receivers_step,
	        true);
    m_f2 << std::endl;

    for (std::size_t p_block_size = 10; p_block_size < std::min(NxS, NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.15) + 15) {
        m_f2 << "inner receivers;";
        m_f2 << data_gen.get_x_dim()*data_gen.get_z_dim() << ";";
        m_f2 << NxR << ";";
        m_f2 << data_gen.get_n_samples() << ";";
        m_f2 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG2DBlocksReceiversInnerLoop<double, double>,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6,
                std::placeholders::_7,
                p_block_size),
                data_gen,
                    m_f2,
                x0_r, x1_r, NxR,
                receivers_step,
                false);
        m_f2 << std::endl;
    }
}

void create_measurements_file(const std::string &filename, std::ofstream& measurements_file) {
     measurements_file.open(filename);
     measurements_file << "summation version;";
     measurements_file << "number of points;";
     measurements_file << "number of receivers;";
     measurements_file << "number of samples;";
     measurements_file << "p block size;";
     for (std::size_t i_e = 0; i_e < Events::COUNT_EVENTS; ++i_e) {
         measurements_file << Events::events_names[i_e] << ";";
     }
     measurements_file << "time, s";
     measurements_file << std::endl;
}

int main(int, char const **) {

    std::ofstream m_files[9];

    std::size_t i_f = 0;
    for (const std::string &filename : {"./m_file_2D_test1_inner_points.csv",
                                        "./m_file_2D_test1_inner_receivers.csv",
                                        "./m_file_2D_test1_strip_mining.csv",
                                        "./m_file_2D_test2_inner_points.csv",
                                        "./m_file_2D_test2_inner_receivers.csv",
                                        "./m_file_2D_test2_strip_mining.csv",
                                        "./m_file_2D_test3_inner_points.csv",
                                        "./m_file_2D_test3_inner_receivers.csv",
                                        "./m_file_2D_test3_strip_mining.csv",}) {
        create_measurements_file(filename, m_files[i_f]);
        ++i_f;
    }

	test_n_sou_greater_n_smpls(m_files[0], m_files[1], m_files[2]);
//	test_n_smpls_greater_n_sou(m_files[3], m_files[4], m_files[5]);
//	test_n_sou_equal_n_smpls(m_files[6], m_files[7], m_files[8]);

	return 0;
}