#ifndef _KIRCHHOFF_MIGRATION_BLOCKS_POINTS_INNER_LOOP_H
#define _KIRCHHOFF_MIGRATION_BLOCKS_POINTS_INNER_LOOP_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <memory>
#include <functional>
#include <iostream>
#include <immintrin.h>

template<typename T1, typename T2>
void kirchhoffMigrationCHG2DBlocksPointsInnerLoop(const Array2D<T1> &gather,
                                                  const std::vector<T2> &times_to_source,
                                                  const Array2D<T2> &times_to_receivers,
                                                  std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                                  double dt,
                                                  T1 *result_data,
                                                  std::ptrdiff_t p_block_size) {

    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                    const std::ptrdiff_t i_z_layer = i_z * x_dim;

                    const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                    _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                    _mm_prefetch(times_to_receivers.get(i_r, i_p_next), _MM_HINT_T0);

                    for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                        const std::ptrdiff_t i_p = i_z_layer + i_x;

                        const T2 t_to_s = times_to_source[i_p];
                        const T2 t_to_r = times_to_receivers(i_r, i_p);

                        const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                        if (sample_idx < n_samples) {
                            result_data[i_p] += gather(i_r, sample_idx);
                        }
                    }
                }
            }
        }
    }

}

template<typename T1, typename T2>
void kirchhoffMigrationCHG2DBlocksPointsInnerLoopStripMining(const Array2D<T1> &gather,
                                                             const std::vector<T2> &times_to_source,
                                                             const Array2D<T2> &times_to_receivers,
                                                             std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                                             double dt,
                                                             T1 *result_data,
                                                             std::ptrdiff_t p_block_size) {

    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();
    const std::ptrdiff_t i_r_last = n_receivers - 1;

    bool if_one_receiver = false;

    if (i_r_last == 0) {
        if_one_receiver = true;
    }

    const double rev_dt = 1.0 / dt;

    if (if_one_receiver) {
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {

                const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                // First receiver processing

                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                    const std::ptrdiff_t i_z_layer = i_z * x_dim;

                    const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                    _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                    _mm_prefetch(times_to_receivers.get(0, i_p_next), _MM_HINT_T0);

                    for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                        const std::ptrdiff_t i_p = i_z_layer + i_x;

                        const T2 t_to_s = times_to_source[i_p];
                        const T2 t_to_r = times_to_receivers(0, i_p);

                        const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                        if (sample_idx < n_samples) {
                            result_data[i_p] += gather(0, sample_idx);
                        }
                    }
                }
            }
        }
    } else {
        std::ptrdiff_t align = 128;
        auto deleter = _mm_free;
        const auto p2_block_size = p_block_size*p_block_size;

        std::unique_ptr<T1[], decltype(deleter)> tmp_result_data{static_cast<T1*>(_mm_malloc(sizeof(T1)*p2_block_size, align)), deleter};

        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {

                const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);
                const std::ptrdiff_t x_block_size = x_block_upper_border - i_b_x;

                // First receiver processing

                const std::ptrdiff_t i_p_block = i_b_z * x_dim + i_b_x;

    //            const auto sample_idx_first_r0 = static_cast<std::ptrdiff_t>(
    //                    (times_to_source[i_p_block] + times_to_receivers(1, i_p_block)) * rev_dt
    //                    );
    //
    //            _mm_prefetch(&gather(1, sample_idx_first_r0), _MM_HINT_T0);

                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                    const std::ptrdiff_t i_z_layer = i_z * x_dim;
                    const std::ptrdiff_t i_z_layer_tmp = (i_z - i_b_z) * x_block_size;

                    const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                    _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                    _mm_prefetch(times_to_receivers.get(0, i_p_next), _MM_HINT_T0);

                    for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                        const std::ptrdiff_t i_p = i_z_layer + i_x;
                        const std::ptrdiff_t i_p_tmp = i_z_layer_tmp + (i_x - i_b_x);

                        const T2 t_to_s = times_to_source[i_p];
                        const T2 t_to_r = times_to_receivers(0, i_p);

                        const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                        if (sample_idx < n_samples) {
                            tmp_result_data[i_p_tmp] = gather(0, sample_idx);
                        }
                    }
                }

                // Second-prelast receivers processing

                for (std::ptrdiff_t i_r = 1; i_r < i_r_last; ++i_r) {

                    const auto sample_idx_first = static_cast<std::ptrdiff_t>(
                            (times_to_source[i_p_block] + times_to_receivers(i_r + 1, i_p_block)) * rev_dt
                    );

                    _mm_prefetch(gather.get(i_r + 1, sample_idx_first), _MM_HINT_T0);

                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                        const std::ptrdiff_t i_z_layer = i_z * x_dim;
                        const std::ptrdiff_t i_z_layer_tmp = (i_z - i_b_z) * x_block_size;

                        const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                        _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                        _mm_prefetch(times_to_receivers.get(i_r, i_p_next), _MM_HINT_T0);

                        for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                            const std::ptrdiff_t i_p = i_z_layer + i_x;
                            const std::ptrdiff_t i_p_tmp = i_z_layer_tmp + (i_x - i_b_x);

                            const T2 t_to_s = times_to_source[i_p];
                            const T2 t_to_r = times_to_receivers(i_r, i_p);

                            const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                            if (sample_idx < n_samples) {
                                tmp_result_data[i_p_tmp] += gather(i_r, sample_idx);
                            }
                        }
                    }
                }

                // Last receiver processing

    //            const auto sample_idx_first_r_last = static_cast<std::ptrdiff_t>(
    //                    (times_to_source[i_p_block] + times_to_receivers(i_r_last, i_p_block)) * rev_dt
    //                    );
    //
    //            _mm_prefetch(&gather(0, sample_idx_first_r_last), _MM_HINT_T0);

                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                    const std::ptrdiff_t i_z_layer = i_z * x_dim;
                    const std::ptrdiff_t i_z_layer_tmp = (i_z - i_b_z) * x_block_size;

                    const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                    _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                    _mm_prefetch(times_to_receivers.get(i_r_last, i_p_next), _MM_HINT_T0);

                    for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                        const std::ptrdiff_t i_p = i_z_layer + i_x;
                        const std::ptrdiff_t i_p_tmp = i_z_layer_tmp + (i_x - i_b_x);

                        result_data[i_p] += tmp_result_data[i_p_tmp];

                        const T2 t_to_s = times_to_source[i_p];
                        const T2 t_to_r = times_to_receivers(i_r_last, i_p);

                        const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                        if (sample_idx < n_samples) {
                            result_data[i_p] += gather(i_r_last, sample_idx);
                        }
                    }
                }
            }
        }
    }

}

template<typename T1, typename T2>
void kirchhoffMigrationCHG3DBlocksPointsInnerLoop(const Array2D<T1> &gather,
                                                  const std::vector<T2> &times_to_source,
                                                  const Array2D<T2> &times_to_receivers,
                                                  std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                                  double dt,
                                                  T1 *result_data,
                                                  std::ptrdiff_t p_block_size) {

    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
                const std::ptrdiff_t y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
                for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                    const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                            const std::ptrdiff_t i_zy_layer = (i_z * y_dim + i_y) * x_dim;

                            const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

                            _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                            _mm_prefetch(times_to_receivers.get(i_r, i_p_next), _MM_HINT_T0);

                            for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                                const std::ptrdiff_t i_p = i_zy_layer + i_x;

                                const T2 t_to_s = times_to_source[i_p];
                                const T2 t_to_r = times_to_receivers(i_r, i_p);

                                const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                                if (sample_idx < n_samples) {
                                    result_data[i_p] += gather(i_r, sample_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

template<typename T1, typename T2>
void kirchhoffMigrationCHG3DBlocksPointsInnerLoopStripMining(const Array2D<T1> &gather,
                                                             const std::vector<T2> &times_to_source,
                                                             const Array2D<T2> &times_to_receivers,
                                                             std::ptrdiff_t z_dim, std::ptrdiff_t y_dim,
                                                             std::ptrdiff_t x_dim,
                                                             double dt,
                                                             T1 *result_data,
                                                             std::ptrdiff_t p_block_size) {

    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();
    const std::ptrdiff_t i_r_last = n_receivers - 1;

    bool if_one_receiver = false;

    if (i_r_last == 0) {
        if_one_receiver = true;
    }

    const double rev_dt = 1.0 / dt;

    if (if_one_receiver) {
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
                const std::ptrdiff_t y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
                for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                    const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                            const std::ptrdiff_t i_zy_layer = (i_z * y_dim + i_y) * x_dim;

                            const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

                            _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                            _mm_prefetch(times_to_receivers.get(0, i_p_next), _MM_HINT_T0);

                            for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                                const std::ptrdiff_t i_p = i_zy_layer + i_x;

                                const T2 t_to_s = times_to_source[i_p];
                                const T2 t_to_r = times_to_receivers(0, i_p);

                                const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                                if (sample_idx < n_samples) {
                                    result_data[i_p] += gather(0, sample_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        const auto p3_block_size = p_block_size*p_block_size*p_block_size;
        std::ptrdiff_t align = 128;
        //    for (; align > p3_block_size; align /= 2) { }
        auto deleter = _mm_free;

        std::unique_ptr<T1[], decltype(deleter)> tmp_result_data{static_cast<T1*>(_mm_malloc(sizeof(T1)*p3_block_size, align)), deleter};

        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
                const std::ptrdiff_t y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
                const std::ptrdiff_t y_block_size = y_block_upper_border - i_b_y;
                for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                    const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);
                    const std::ptrdiff_t x_block_size = x_block_upper_border - i_b_x;

    //                const std::ptrdiff_t i_p_block = (i_b_z * y_dim + i_b_y) * x_dim + i_b_x;

    //                const auto sample_idx_first_r0 = static_cast<std::ptrdiff_t>(
    //                        (times_to_source[i_p_block] + times_to_receivers(0, i_p_block)) * rev_dt
    //                );
    //
    //                if (sample_idx_first_r0 < n_samples) {
    //                    _mm_prefetch(&gather(0, sample_idx_first_r0), _MM_HINT_T0);
    //                }

                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        const std::ptrdiff_t i_z_tmp = (i_z - i_b_z);
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                            const std::ptrdiff_t i_zy_layer = (i_z * y_dim + i_y) * x_dim;
                            const std::ptrdiff_t i_zy_layer_tmp =
                                    (i_z_tmp * y_block_size + (i_y - i_b_y)) * x_block_size;

                            const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

                            _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                            _mm_prefetch(times_to_receivers.get(0, i_p_next), _MM_HINT_T0);

                            for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                                const std::ptrdiff_t i_p = i_zy_layer + i_x;
                                const std::ptrdiff_t i_p_tmp = i_zy_layer_tmp + (i_x - i_b_x);

                                const T2 t_to_s = times_to_source[i_p];
                                const T2 t_to_r = times_to_receivers(0, i_p);

                                const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                                if (sample_idx < n_samples) {
                                    tmp_result_data[i_p_tmp] = gather(0, sample_idx);
                                }
                            }
                        }
                    }

                    for (std::ptrdiff_t i_r = 1; i_r < i_r_last; ++i_r) {

    //                    const auto sample_idx_first = static_cast<std::ptrdiff_t>(
    //                            (times_to_source[i_p_block] + times_to_receivers(i_r + 1, i_p_block)) * rev_dt
    //                    );
    //
    //                    _mm_prefetch(gather.get(i_r + 1, sample_idx_first), _MM_HINT_T0);

                        for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                            const std::ptrdiff_t i_z_tmp = (i_z - i_b_z);
                            for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                                const std::ptrdiff_t i_zy_layer = (i_z * y_dim + i_y) * x_dim;
                                const std::ptrdiff_t i_zy_layer_tmp =
                                        (i_z_tmp * y_block_size + (i_y - i_b_y)) * x_block_size;

                                const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

                                _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                                _mm_prefetch(times_to_receivers.get(i_r, i_p_next), _MM_HINT_T0);

                                for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                                    const std::ptrdiff_t i_p = i_zy_layer + i_x;
                                    const std::ptrdiff_t i_p_tmp = i_zy_layer_tmp + (i_x - i_b_x);

                                    const T2 t_to_s = times_to_source[i_p];
                                    const T2 t_to_r = times_to_receivers(i_r, i_p);

                                    const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                                    if (sample_idx < n_samples) {
                                        tmp_result_data[i_p_tmp] += gather(i_r, sample_idx);
                                    }
                                }
                            }
                        }
                    }

    //                const auto sample_idx_first_r_last = static_cast<std::ptrdiff_t>(
    //                        (times_to_source[i_p_block] + times_to_receivers(i_r_last, i_p_block)) * rev_dt
    //                );
    //
    //                if (sample_idx_first_r_last < n_samples) {
    //                    _mm_prefetch(&gather(i_r_last, sample_idx_first_r_last), _MM_HINT_T0);
    //                }

                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        const std::ptrdiff_t i_z_tmp = (i_z - i_b_z);
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                            const std::ptrdiff_t i_zy_layer = (i_z * y_dim + i_y) * x_dim;
                            const std::ptrdiff_t i_zy_layer_tmp =
                                    (i_z_tmp * y_block_size + (i_y - i_b_y)) * x_block_size;

                            const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

                            _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                            _mm_prefetch(times_to_receivers.get(i_r_last, i_p_next), _MM_HINT_T0);

                            for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                                const std::ptrdiff_t i_p = i_zy_layer + i_x;
                                const std::ptrdiff_t i_p_tmp = i_zy_layer_tmp + (i_x - i_b_x);

                                result_data[i_p] += tmp_result_data[i_p_tmp];

                                const T2 t_to_s = times_to_source[i_p];
                                const T2 t_to_r = times_to_receivers(i_r_last, i_p);

                                const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                                if (sample_idx < n_samples) {
                                    result_data[i_p] += gather(i_r_last, sample_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

#endif //_KIRCHHOFF_MIGRATION_BLOCKS_POINTS_INNER_LOOP_H