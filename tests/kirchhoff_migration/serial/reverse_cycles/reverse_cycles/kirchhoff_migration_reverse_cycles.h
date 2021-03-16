#ifndef _KIRCHHOFF_MIGRATION_REVERSE_CYCLES_H
#define _KIRCHHOFF_MIGRATION_REVERSE_CYCLES_H

#include "array2D.h"
#include "arrival_times_calculator2D.h"

#include "array3D.h"
#include "arrival_times_calculator3D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <numeric>

template <typename T1, typename T2, typename CalculatorType, typename T>
void kirchhoffMigrationCHG2DReverseCycles(const Array2D<T1> &gather,
                                const std::vector<T2> &receivers_coords,
                                double s_x,
                                double dt,
                                ArrivalTimesCalculator2D<CalculatorType, T> &arrival_times_calculator,
                                T1 *result_data) {
    
    std::ptrdiff_t z_dim = arrival_times_calculator.get_z_dim(), x_dim = arrival_times_calculator.get_x_dim();
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    const std::vector<std::pair<double, double>> &grid = arrival_times_calculator.get_grid();

    double z0 = grid[0].first, z1 = grid[0].second;
    double dz = (z1 - z0) / (z_dim - 1);
    double x0 = grid[1].first, x1 = grid[1].second;
    double dx = (x1 - x0) / (x_dim - 1);

    std::vector<double> s_r_x_v{2, s_x};

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
        s_r_x_v[1] = receivers_coords[i_r];

        for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
            std::vector<double> p_z_v{2, z0 + i_z*dz};

            for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {

                std::vector<double> p_x_v{2, x0 + i_x*dx};

                auto t_to_s_r = arrival_times_calculator(s_r_x_v, p_z_v, p_x_v);

                std::ptrdiff_t sample_idx = static_cast<std::ptrdiff_t>((t_to_s_r[0] + t_to_s_r[1]) / dt);
                
                if (sample_idx < n_samples) {
                    result_data[i_z*x_dim + i_x] += gather(i_r, sample_idx);
                }

            }
        }

    }
}

template <typename T1, typename T2, typename CalculatorType, typename T>
void kirchhoffMigrationCHG3DReverseCycles(const Array2D<T1> &gather,
                                const Array2D<T2> &receivers_coords,
                                double s_x, double s_y,
                                double dt,
                                ArrivalTimesCalculator3D<CalculatorType, T> &arrival_times_calculator,
                                T1 *result_data) {
    
    std::ptrdiff_t z_dim = arrival_times_calculator.get_z_dim(), 
                    y_dim = arrival_times_calculator.get_y_dim(), 
                    x_dim = arrival_times_calculator.get_x_dim();
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    const std::vector<std::pair<double, double>> &grid = arrival_times_calculator.get_grid();

    double z0 = grid[0].first, z1 = grid[0].second;
    double dz = (z1 - z0) / (z_dim - 1);
    double y0 = grid[1].first, y1 = grid[1].second;
    double dy = (y1 - y0) / (y_dim - 1);
    double x0 = grid[2].first, x1 = grid[2].second;
    double dx = (x1 - x0) / (x_dim - 1);

    std::vector<double> s_r_x_v{2, s_x}, s_r_y_v{2, s_y};

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
        s_r_x_v[1] = receivers_coords(i_r, 0), s_r_y_v[1] = receivers_coords(i_r, 1);

        for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
            std::vector<double> p_z_v{2, z0 + i_z*dz};

            for (std::ptrdiff_t i_y = 0; i_y < y_dim; ++i_y) {
                std::vector<double> p_y_v{2, y0 + i_y*dy};

                for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {

                    std::vector<double> p_x_v{2, x0 + i_x*dx};

                    auto t_to_s_r = arrival_times_calculator(s_r_x_v, s_r_y_v, p_z_v, p_y_v, p_x_v);

                    std::ptrdiff_t sample_idx = static_cast<std::ptrdiff_t>((t_to_s_r[0] + t_to_s_r[1]) / dt);
                    
                    if (sample_idx < n_samples) {
                        result_data[(i_z*y_dim + i_y)*x_dim + i_x] += gather(i_r, sample_idx);
                    }

                }
            }
        }

    }
}

#endif //_KIRCHHOFF_MIGRATION_REVERSE_CYCLES_H