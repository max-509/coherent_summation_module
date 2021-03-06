#include "kirchhoff_migration_reverse_cycles.h"
#include "kirchhoff_migration_blocks_receivers_inner_loop.h"
#include "kirchhoff_migration_blocks_points_inner_loop.h"
#include "kirchhoff_migration_native.h"
#include "perf_wrapper.h"
#include "test_data_generator3D.h"

#include <functional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>
#include <initializer_list>

template<typename T1, typename T2>
using CohSumType = std::function<void(const Array2D<T1> &,
                                      const std::vector<T2> &,
                                      const Array2D<T2> &,
                                      std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t,
                                      double,
                                      T1 *)>;

void run_program(CohSumType<double, double> coh_sum, test_data_generator3D<double> &data_gen,
                 std::ofstream &measurements_file,
                 double x0_r, double x1_r, std::size_t NxR,
                 double y0_r, double y1_r, std::size_t NyR,
                 std::size_t receivers_step,
                 bool is_trans) {
    auto times_to_source = data_gen.get_times_to_source();
    std::ptrdiff_t z_dim = data_gen.get_z_dim(), y_dim = data_gen.get_y_dim(), x_dim = data_gen.get_x_dim();
    double dt = data_gen.get_dt();

    double *result_data = new double[z_dim * y_dim * x_dim];

    double dx_r = (x1_r - x0_r) / (NxR - 1);
    double dy_r = (y1_r - y0_r) / (NyR - 1);

    std::vector<double> receivers_coords(NxR * NyR * 2);

    std::vector<uint64_t> events_counts(Events::COUNT_EVENTS, 0);

    double time_in_sec = 0.0;

    for (std::size_t i_r_y = 0; i_r_y < NyR; ++i_r_y) {
        double y_r = y0_r + dy_r * i_r_y;
        for (std::size_t i_r_x = 0; i_r_x < NxR; ++i_r_x) {
            double x_r = x0_r + dx_r * i_r_x;
            receivers_coords[(i_r_y * NxR + i_r_x) * 2 + 0] = y_r;
            receivers_coords[(i_r_y * NxR + i_r_x) * 2 + 1] = x_r;
        }
    }

    for (std::ptrdiff_t rec_bl = 0; rec_bl < NxR * NyR; rec_bl += receivers_step) {
        std::ptrdiff_t upper_border_receivers_block = std::min(NxR * NyR, rec_bl + receivers_step);
        std::vector<std::pair<double, double>> receivers_coords_block(upper_border_receivers_block - rec_bl);
        for (std::size_t i_r = rec_bl; i_r < upper_border_receivers_block; ++i_r) {
            receivers_coords_block[i_r - rec_bl].first = receivers_coords[(i_r) * 2];
            receivers_coords_block[i_r - rec_bl].second = receivers_coords[(i_r) * 2 + 1];
        }

        auto user_datas = data_gen.generate_user_data_by_receivers<double>(receivers_coords_block, is_trans);

        Array2D<double> gather(user_datas.second, upper_border_receivers_block - rec_bl, data_gen.get_n_samples());

        std::pair<double, std::vector<uint64_t>> res;
        if (is_trans) {
            Array2D<double> times_to_receivers(user_datas.first.get(), upper_border_receivers_block - rec_bl,
                                               z_dim * y_dim * x_dim);
            res = perf_wrapper(
                    std::bind(coh_sum, std::ref(gather), std::ref(times_to_source), std::ref(times_to_receivers), z_dim,
                              y_dim, x_dim, dt, result_data));
        } else {
            Array2D<double> times_to_receivers(user_datas.first.get(), z_dim * y_dim * x_dim,
                                               upper_border_receivers_block - rec_bl);
            res = perf_wrapper(
                    std::bind(coh_sum, std::ref(gather), std::ref(times_to_source), std::ref(times_to_receivers), z_dim,
                              y_dim, x_dim, dt, result_data));
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

    delete[] result_data;

}

void test_n_sou_greater_n_smpls(std::ofstream &m_f1, std::ofstream &m_f2, std::ofstream &m_f3) {
    double x0_r = 0, x1_r = 4000;
    std::size_t NxR = 40;
    double y0_r = 0, y1_r = 4000;
    std::size_t NyR = 40;
    std::size_t receivers_step = 20;
    double x0_s = 0, x1_s = 4000;
    std::size_t NxS = 420;
    double y0_s = 0, y1_s = 4000;
    std::size_t NyS = 420;
    double z0_s = 0, z1_s = 2000;
    std::size_t NzS = 420;
    std::size_t n_samples = 20000;
    double velocity = 3500.0;
    double s_x = 0.0, s_y = 0.0;
    double dt = 0.002;

    test_data_generator3D<double> data_gen(x0_s, x1_s, NxS,
                                           y0_s, y1_s, NyS,
                                           z0_s, z1_s, NzS,
                                           s_x, s_y, dt,
                                           n_samples,
                                           velocity);

//	m_f1 << "reverse_cycles;";
//	m_f1 << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
//    m_f1 << NxR*NyR << ";";
//    m_f1 << data_gen.get_n_samples() << ";";
//    m_f1 << 0 << ";";
//	run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
//	        data_gen,
//	        m_f1,
//	        x0_r, x1_r, NxR,
//	        y0_r, y1_r, NyR,
//	        receivers_step,
//	        true);
//    m_f1 << std::endl;

    m_f1 << "native;";
    m_f1 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f1 << NxR * NyR << ";";
    m_f1 << data_gen.get_n_samples() << ";";
    m_f1 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DNative < double, double > ,
                data_gen,
                m_f1,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                false);
    m_f1 << std::endl;

//	for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.1) + 9) {
//        m_f1 << "inner points;";
//        m_f1 << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
//        m_f1 << NxR*NyR << ";";
//        m_f1 << data_gen.get_n_samples() << ";";
//        m_f1 << p_block_size << ";";
//        run_program(std::bind(kirchhoffMigrationCHG3DBlocksPointsInnerLoop<double, double>,
//                std::placeholders::_1,
//                std::placeholders::_2,
//                std::placeholders::_3,
//                std::placeholders::_4,
//                std::placeholders::_5,
//                std::placeholders::_6,
//                std::placeholders::_7,
//                std::placeholders::_8,
//                p_block_size),
//                data_gen,
//                    m_f1,
//                x0_r, x1_r, NxR,
//	            y0_r, y1_r, NyR,
//                receivers_step,
//                true);
//        m_f1 << std::endl;
//    }
//
//	m_f3 << "reverse_cycles;";
//	m_f3 << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
//    m_f3 << NxR*NyR << ";";
//    m_f3 << data_gen.get_n_samples() << ";";
//    m_f3 << 0 << ";";
//	run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
//	        data_gen,
//	        m_f3,
//	        x0_r, x1_r, NxR,
//	        y0_r, y1_r, NyR,
//	        receivers_step,
//	        true);
//    m_f3 << std::endl;
//
//	for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size = static_cast<std::size_t>(p_block_size * 1.1) + 9) {
//        m_f3 << "strip mining;";
//        m_f3 << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
//        m_f3 << NxR*NyR << ";";
//        m_f3 << data_gen.get_n_samples() << ";";
//        m_f3 << p_block_size << ";";
//        run_program(std::bind(kirchhoffMigrationCHG3DBlocksPointsInnerLoopStripMining<double, double>,
//                std::placeholders::_1,
//                std::placeholders::_2,
//                std::placeholders::_3,
//                std::placeholders::_4,
//                std::placeholders::_5,
//                std::placeholders::_6,
//                std::placeholders::_7,
//                std::placeholders::_8,
//                p_block_size),
//                data_gen,
//                    m_f3,
//                x0_r, x1_r, NxR,
//	            y0_r, y1_r, NyR,
//                receivers_step,
//                true);
//        m_f3 << std::endl;
//    }
//
//	m_f2 << "reverse_cycles;";
//	m_f2 << data_gen.get_x_dim()*data_gen.get_y_dim()*data_gen.get_z_dim() << ";";
//    m_f2 << NxR*NyR << ";";
//    m_f2 << data_gen.get_n_samples() << ";";
//    m_f2 << 0 << ";";
//	run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
//	        data_gen,
//	        m_f2,
//	        x0_r, x1_r, NxR,
//	        y0_r, y1_r, NyR,
//	        receivers_step,
//	        true);
//    m_f2 << std::endl;

    m_f2 << "inner receivers;";
    m_f2 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f2 << NxR * NyR << ";";
    m_f2 << data_gen.get_n_samples() << ";";
    m_f2 << 120 << ";";
    run_program(std::bind(kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double>,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3,
                          std::placeholders::_4,
                          std::placeholders::_5,
                          std::placeholders::_6,
                          std::placeholders::_7,
                          std::placeholders::_8,
                          120),
                data_gen,
                m_f2,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                false);
    m_f2 << std::endl;

//    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
//                                                                                                 static_cast<std::size_t>(
//                                                                                                         p_block_size *
//                                                                                                         1.1) + 9) {
//        m_f2 << "inner receivers;";
//        m_f2 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
//        m_f2 << NxR * NyR << ";";
//        m_f2 << data_gen.get_n_samples() << ";";
//        m_f2 << p_block_size << ";";
//        run_program(std::bind(kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double>,
//                              std::placeholders::_1,
//                              std::placeholders::_2,
//                              std::placeholders::_3,
//                              std::placeholders::_4,
//                              std::placeholders::_5,
//                              std::placeholders::_6,
//                              std::placeholders::_7,
//                              std::placeholders::_8,
//                              p_block_size),
//                    data_gen,
//                    m_f2,
//                    x0_r, x1_r, NxR,
//                    y0_r, y1_r, NyR,
//                    receivers_step,
//                    false);
//        m_f2 << std::endl;
//    }
}

void test_n_smpls_greater_n_sou(std::ofstream &m_f1, std::ofstream &m_f2, std::ofstream &m_f3) {
    double x0_r = 0, x1_r = 4000;
    std::size_t NxR = 54;
    double y0_r = 0, y1_r = 4000;
    std::size_t NyR = 54;
    std::size_t receivers_step = 50;
    double x0_s = 0, x1_s = 4000;
    std::size_t NxS = 185;
    double y0_s = 0, y1_s = 4000;
    std::size_t NyS = 185;
    double z0_s = 0, z1_s = 2000;
    std::size_t NzS = 185;
    std::size_t n_samples = 10000000;
    double velocity = 3500.0;
    double s_x = 0.0, s_y = 0.0;
    double dt = 0.002;

    test_data_generator3D<double> data_gen(x0_s, x1_s, NxS,
                                           y0_s, y1_s, NyS,
                                           z0_s, z1_s, NzS,
                                           s_x, s_y, dt,
                                           n_samples,
                                           velocity);

    m_f1 << "reverse_cycles;";
    m_f1 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f1 << NxR * NyR << ";";
    m_f1 << data_gen.get_n_samples() << ";";
    m_f1 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
                data_gen,
                m_f1,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                true);
    m_f1 << std::endl;

    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
                                                                                                 static_cast<std::size_t>(
                                                                                                         p_block_size *
                                                                                                         1.1) + 9) {
        m_f1 << "inner points;";
        m_f1 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
        m_f1 << NxR * NyR << ";";
        m_f1 << data_gen.get_n_samples() << ";";
        m_f1 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG3DBlocksPointsInnerLoop<double, double>,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              std::placeholders::_4,
                              std::placeholders::_5,
                              std::placeholders::_6,
                              std::placeholders::_7,
                              std::placeholders::_8,
                              p_block_size),
                    data_gen,
                    m_f1,
                    x0_r, x1_r, NxR,
                    y0_r, y1_r, NyR,
                    receivers_step,
                    true);
        m_f1 << std::endl;
    }

    m_f3 << "reverse_cycles;";
    m_f3 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f3 << NxR * NyR << ";";
    m_f3 << data_gen.get_n_samples() << ";";
    m_f3 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
                data_gen,
                m_f3,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                true);
    m_f3 << std::endl;

    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
                                                                                                 static_cast<std::size_t>(
                                                                                                         p_block_size *
                                                                                                         1.1) + 9) {
        m_f3 << "strip mining;";
        m_f3 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
        m_f3 << NxR * NyR << ";";
        m_f3 << data_gen.get_n_samples() << ";";
        m_f3 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG3DBlocksPointsInnerLoopStripMining<double, double>,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              std::placeholders::_4,
                              std::placeholders::_5,
                              std::placeholders::_6,
                              std::placeholders::_7,
                              std::placeholders::_8,
                              p_block_size),
                    data_gen,
                    m_f3,
                    x0_r, x1_r, NxR,
                    y0_r, y1_r, NyR,
                    receivers_step,
                    true);
        m_f3 << std::endl;
    }

    m_f2 << "reverse_cycles;";
    m_f2 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f2 << NxR * NyR << ";";
    m_f2 << data_gen.get_n_samples() << ";";
    m_f2 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
                data_gen,
                m_f2,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                true);
    m_f2 << std::endl;

    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
                                                                                                 static_cast<std::size_t>(
                                                                                                         p_block_size *
                                                                                                         1.1) + 9) {
        m_f2 << "inner receivers;";
        m_f2 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
        m_f2 << NxR * NyR << ";";
        m_f2 << data_gen.get_n_samples() << ";";
        m_f2 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double>,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              std::placeholders::_4,
                              std::placeholders::_5,
                              std::placeholders::_6,
                              std::placeholders::_7,
                              std::placeholders::_8,
                              p_block_size),
                    data_gen,
                    m_f2,
                    x0_r, x1_r, NxR,
                    y0_r, y1_r, NyR,
                    receivers_step,
                    false);
        m_f2 << std::endl;
    }


}

void test_n_sou_equal_n_smpls(std::ofstream &m_f1, std::ofstream &m_f2, std::ofstream &m_f3) {
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
    std::size_t n_samples = 15625000;
    double velocity = 3500.0;
    double s_x = 0.0, s_y = 0.0;
    double dt = 0.002;

    test_data_generator3D<double> data_gen(x0_s, x1_s, NxS,
                                           y0_s, y1_s, NyS,
                                           z0_s, z1_s, NzS,
                                           s_x, s_y, dt,
                                           n_samples,
                                           velocity);

    m_f1 << "reverse_cycles;";
    m_f1 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f1 << NxR * NyR << ";";
    m_f1 << data_gen.get_n_samples() << ";";
    m_f1 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
                data_gen,
                m_f1,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                true);
    m_f1 << std::endl;

    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
                                                                                                 static_cast<std::size_t>(
                                                                                                         p_block_size *
                                                                                                         1.1) + 9) {
        m_f1 << "inner points;";
        m_f1 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
        m_f1 << NxR * NyR << ";";
        m_f1 << data_gen.get_n_samples() << ";";
        m_f1 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG3DBlocksPointsInnerLoop<double, double>,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              std::placeholders::_4,
                              std::placeholders::_5,
                              std::placeholders::_6,
                              std::placeholders::_7,
                              std::placeholders::_8,
                              p_block_size),
                    data_gen,
                    m_f1,
                    x0_r, x1_r, NxR,
                    y0_r, y1_r, NyR,
                    receivers_step,
                    true);
        m_f1 << std::endl;
    }

    m_f3 << "reverse_cycles;";
    m_f3 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f3 << NxR * NyR << ";";
    m_f3 << data_gen.get_n_samples() << ";";
    m_f3 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
                data_gen,
                m_f3,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                true);
    m_f3 << std::endl;

    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
                                                                                                 static_cast<std::size_t>(
                                                                                                         p_block_size *
                                                                                                         1.1) + 9) {
        m_f3 << "strip mining;";
        m_f3 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
        m_f3 << NxR * NyR << ";";
        m_f3 << data_gen.get_n_samples() << ";";
        m_f3 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG3DBlocksPointsInnerLoopStripMining<double, double>,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              std::placeholders::_4,
                              std::placeholders::_5,
                              std::placeholders::_6,
                              std::placeholders::_7,
                              std::placeholders::_8,
                              p_block_size),
                    data_gen,
                    m_f3,
                    x0_r, x1_r, NxR,
                    y0_r, y1_r, NyR,
                    receivers_step,
                    true);
        m_f3 << std::endl;
    }

    m_f2 << "reverse_cycles;";
    m_f2 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
    m_f2 << NxR * NyR << ";";
    m_f2 << data_gen.get_n_samples() << ";";
    m_f2 << 0 << ";";
    run_program(kirchhoffMigrationCHG3DReverseCycles<double, double>,
                data_gen,
                m_f2,
                x0_r, x1_r, NxR,
                y0_r, y1_r, NyR,
                receivers_step,
                true);
    m_f2 << std::endl;

    for (std::size_t p_block_size = 5; p_block_size < std::min(std::min(NxS, NyS), NzS); p_block_size =
                                                                                                 static_cast<std::size_t>(
                                                                                                         p_block_size *
                                                                                                         1.1) + 9) {
        m_f2 << "inner receivers;";
        m_f2 << data_gen.get_x_dim() * data_gen.get_y_dim() * data_gen.get_z_dim() << ";";
        m_f2 << NxR * NyR << ";";
        m_f2 << data_gen.get_n_samples() << ";";
        m_f2 << p_block_size << ";";
        run_program(std::bind(kirchhoffMigrationCHG3DBlocksReceiversInnerLoop<double, double>,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              std::placeholders::_4,
                              std::placeholders::_5,
                              std::placeholders::_6,
                              std::placeholders::_7,
                              std::placeholders::_8,
                              p_block_size),
                    data_gen,
                    m_f2,
                    x0_r, x1_r, NxR,
                    y0_r, y1_r, NyR,
                    receivers_step,
                    false);
        m_f2 << std::endl;
    }
}

void create_measurements_file(const std::string &filename, std::ofstream &measurements_file) {
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

int main(int argc, char const *argv[]) {

    std::ofstream m_files[9];

    std::size_t i_f = 0;
    for (const std::string &filename : {"./m_file_3D_test1_inner_points.csv",
                                        "./m_file_3D_test1_inner_receivers.csv",
                                        "./m_file_3D_test1_strip_mining.csv",
                                        "./m_file_3D_test2_inner_points.csv",
                                        "./m_file_3D_test2_inner_receivers.csv",
                                        "./m_file_3D_test2_strip_mining.csv",
                                        "./m_file_3D_test3_inner_points.csv",
                                        "./m_file_3D_test3_inner_receivers.csv",
                                        "./m_file_3D_test3_strip_mining.csv",}) {
        create_measurements_file(filename, m_files[i_f]);
        ++i_f;
    }

    test_n_sou_greater_n_smpls(m_files[0], m_files[1], m_files[2]);
//	test_n_smpls_greater_n_sou(m_files[3], m_files[4], m_files[5]);
//	test_n_sou_equal_n_smpls(m_files[6], m_files[7], m_files[8]);

    return 0;
}