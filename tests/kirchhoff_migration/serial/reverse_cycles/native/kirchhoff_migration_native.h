#ifndef _KIRCHHOFF_MIGRATION_NATIVE_H
#define _KIRCHHOFF_MIGRATION_NATIVE_H

#include "array2D.h"
#include "arrival_times_calculator2D.h"

#include "array3D.h"
#include "arrival_times_calculator3D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>

template <typename T1, typename T2, typename CalculatorType, typename T>
void kirchhoffMigrationCHG2DNative(const Array2D<T1> &gather,
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

    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
        double p_z = z0 + i_z*dz;
        for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {
            T1 result_sum = static_cast<T1>(0.0);

            double p_x = x0 + i_x*dx;

            double t_to_s = arrival_times_calculator(s_x, p_z, p_x);

            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                double r_x = receivers_coords[i_r];

                double t_to_r = arrival_times_calculator(r_x, p_z, p_x);

                std::ptrdiff_t sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) / dt);
                if (sample_idx < n_samples) {
                    result_sum += gather(i_r, sample_idx);
                }
            }

            result_data[i_z*x_dim + i_x] = result_sum;
        }
    }
}

template <typename T1, typename T2, typename CalculatorType, typename T>
void kirchhoffMigrationCHG3DNative(const Array2D<T1> &gather,
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

    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
        double p_z = z0 + i_z*dz;
        for (std::ptrdiff_t i_y = 0; i_y< y_dim; ++i_y) {
            double p_y = y0 + i_y*dy;
            for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {
                T1 result_sum = static_cast<T1>(0.0);

                double p_x = x0 + i_x*dx;

                double t_to_s = arrival_times_calculator(s_x, s_y, p_z, p_y, p_x);

                for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                    double r_x = receivers_coords(i_r, 0), r_y = receivers_coords(i_r, 1);

                    double t_to_r = arrival_times_calculator(r_x, r_y, p_z, p_y, p_x);

                    std::ptrdiff_t sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) / dt);
                    if (sample_idx < n_samples) {
                        result_sum += gather(i_r, sample_idx);
                    }
                }

                result_data[(i_z*y_dim + i_y)*x_dim + i_x] = result_sum;
            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_NATIVE_H