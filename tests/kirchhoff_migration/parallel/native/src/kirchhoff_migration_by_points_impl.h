#ifndef KIRCHHOFF_MIGRATION_BY_POINTS_IMPL_H_
#define KIRCHHOFF_MIGRATION_BY_POINTS_IMPL_H_

#define ENABLED_SIMD_EXTENSIONS

#ifdef __AVX512F__ // AVX512 Enabled

#include "avx512_functions_impl.h"

using SimdFunctionsImpl = Avx512FunctionsImpl;
using vector_pd_t = __m512d;
using vector_ps_t = __m512;

#elif __AVX2__ // AVX2 Enabled

#include "avx2_functions_impl.h"

using SimdFunctionsImpl = Avx2FunctionsImpl;
using vector_pd_t = __m256d;
using vector_ps_t = __m256;

#elif __SSE4_2__ // SSE4.2 Enabled

#include "sse4_functions_impl.h"

using SimdFunctionsImpl = Sse4FunctionsImpl;
using vector_pd_t = __m128d;
using vector_ps_t = __m128;

#else //Without SIMD x86 extensions
#undef ENABLED_SIMD_EXTENSIONS
#endif //Selecting SIMD extensions

#include <cstdint>
#include <type_traits>
#include <memory>
#include <vector>

#include "simd_functions.h"
#include "array2D.h"

template<std::size_t i>
struct ForLoopResTmp {
    template<typename T1, typename Func>
    inline static void assign_res_tmp(const Func &f,
                                      const std::ptrdiff_t i_p,
                                      T1 *res_sum_arr) {
        ForLoopResTmp<i - 1>::assign_res_tmp(f, i_p, res_sum_arr);

        res_sum_arr[i_p + i] = f(i_p + i);
    }
};

template<>
struct ForLoopResTmp<0> {
    template<typename T1, typename Func>
    inline static void assign_res_tmp(const Func &f,
                                      const std::ptrdiff_t i_p,
                                      T1 *res_sum_arr) {
        res_sum_arr[i_p] = f(i_p);
    }
};

template<std::size_t i>
struct ForSumIfMaskInnerReceivers {
    template<typename T1, typename T2, typename mask_t>
    inline static T1 sum_if_mask(const Array2D<T1> &gather,
                                 const T2 *i_samples,
                                 const mask_t i_samples_mask,
                                 const std::ptrdiff_t i_r) {

        T1 res_tmp = ForSumIfMaskInnerReceivers<i - 1>::sum_if_mask(gather, i_samples, i_samples_mask, i_r);

        constexpr auto mask_i = (0x1 << i);
        if (i_samples_mask & mask_i) {
            res_tmp += gather(i_r + i, static_cast<std::ptrdiff_t>(i_samples[i]));
        }

        return res_tmp;
    }
};

template<>
struct ForSumIfMaskInnerReceivers<0> {
    template<typename T1, typename T2, typename mask_t>
    inline static T1 sum_if_mask(const Array2D<T1> &gather,
                                 const T2 *i_samples,
                                 const mask_t i_samples_mask,
                                 const std::ptrdiff_t i_r) {

        T1 res_tmp = T1(0.0);
        if (i_samples_mask & 0x1) {
            res_tmp += gather(i_r, static_cast<std::ptrdiff_t>(i_samples[0]));
        }

        return res_tmp;
    }
};

template<typename T1, typename T2>
inline void process_receivers_on_points(const Array2D<T1> &gather,
                                        const std::vector<T2> &times_to_source,
                                        const Array2D<T2> &times_to_receivers,
                                        const std::ptrdiff_t i_p0,
                                        const std::ptrdiff_t i_pn,
                                        const double rev_dt,
                                        T1 *result_data) {

    const auto n_samples = gather.get_y_dim();
    const auto n_receivers = gather.get_y_dim();

#ifdef ENABLED_SIMD_EXTENSIONS

    using T1_without_cv = typename std::remove_cv<T1>::type;
    using T2_without_cv = typename std::remove_cv<T2>::type;

    using vector_data_t = typename std::conditional<std::is_same<T1_without_cv, double>::value, vector_pd_t, vector_ps_t>::type;
    using vector_arrival_time_t = typename std::conditional<std::is_same<T2_without_cv, double>::value, vector_pd_t, vector_ps_t>::type;

    static const SimdFunctions<SimdFunctionsImpl> simd_functions{SimdFunctionsImpl{}};

    constexpr std::ptrdiff_t vector_dim_arrival_times = sizeof(vector_arrival_time_t) / sizeof(T2);
    constexpr std::ptrdiff_t vector_dim_data = sizeof(vector_data_t) / sizeof(T1);

    std::ptrdiff_t i_p = (i_p0 + vector_dim_data - 1) & ~(vector_dim_data - 1);

    const std::ptrdiff_t n_receivers_without_remainder = n_receivers - (n_receivers % vector_dim_arrival_times);
    const std::ptrdiff_t n_points_without_remainder = i_pn - (i_pn % vector_dim_data);

    alignas(sizeof(vector_arrival_time_t)) T2_without_cv i_samples[vector_dim_arrival_times];
//    alignas(sizeof(vector_data_t)) T1_without_cv res_tmp_arr[vector_dim_data];

    const auto rev_dt_v = simd_functions.set1(static_cast<T2>(rev_dt));
    const auto n_samples_v = simd_functions.set1(static_cast<T2>(n_samples));

    auto point_processer = [&gather,
            &times_to_source,
            &times_to_receivers,
            vector_dim_data,
            vector_dim_arrival_times,
            n_receivers,
            n_receivers_without_remainder,
            &i_samples,
            rev_dt,
            rev_dt_v,
            n_samples,
            n_samples_v] (const std::ptrdiff_t i_p) -> T1 {

        const auto t_to_s = times_to_source[i_p];
        const auto t_to_s_v = simd_functions.set1(t_to_s);
        auto res_sum = T1(0.0);

        for (auto i_r = 0; i_r < n_receivers_without_remainder; i_r += vector_dim_arrival_times) {
            const auto t_to_r_v = simd_functions.loadu(times_to_receivers.get(i_p, i_r));

            auto v_i_sample_f = simd_functions.mul(rev_dt_v, simd_functions.add(t_to_r_v, t_to_s_v));

            simd_functions.storeu(i_samples, v_i_sample_f);
            auto i_sample_mask = simd_functions.movemask(simd_functions.cmpgt(n_samples_v, v_i_sample_f));

            res_sum += ForSumIfMaskInnerReceivers<vector_dim_arrival_times - 1>::sum_if_mask(gather, i_samples, i_sample_mask, i_r);
        }

        res_sum += process_remainder_receivers(gather, t_to_s, times_to_receivers, i_p, n_receivers_without_remainder,
                                               n_receivers, n_samples, rev_dt);

        return res_sum;
    };

    #pragma omp parallel for schedule(static)
    for (auto i = i_p0; i < i_p; ++i) {
        result_data[i] += point_processer(i);
    }

    #pragma omp parallel for schedule(static)
    for (auto i = i_p; i < n_points_without_remainder; i += vector_dim_data) {

        ForLoopResTmp<vector_dim_data - 1>::assign_res_tmp(point_processer, i, result_data);

    }

    #pragma omp parallel for schedule(static)
    for (auto i = n_points_without_remainder; i < i_pn; ++i) {
        result_data[i] += point_processer(i);
    }

#else //ENABLED_SIMD_EXTENSIONS

    #pragma omp parallel for schedule(static)
    for (auto i_p = i_p0; i_p < i_pn; ++i_p) {
        auto res_sum = T1(0.0);
        const auto t_to_s = times_to_source[i_p];

        #pragma omp simd
        for (auto i_r = 0; i_r < n_receivers; ++i_r) {
            const auto t_to_r = times_to_receivers(i_p, i_r);

            const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

            if (sample_idx < n_samples) {
                res_sum += gather(i_r, sample_idx);
            }
        }

        result_data[i_p] += res_sum;
    }

#endif //ENABLED_SIMD_EXTENSIONS

}

template<typename T1, typename T2>
inline T1 process_remainder_receivers(const Array2D<T1> &gather,
                                      const T2 t_to_s,
                                      const Array2D<T2> &times_to_receivers,
                                      const std::ptrdiff_t i_p,
                                      const std::ptrdiff_t i_r0,
                                      const std::ptrdiff_t i_rn,
                                      const std::ptrdiff_t n_samples,
                                      const double rev_dt) {

    auto res_sum = T1(0.0);

    #pragma omp simd
    for (auto i_r = i_r0; i_r < i_rn; ++i_r) {
        const auto t_to_r = times_to_receivers(i_p, i_r);

        const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

        if (sample_idx < n_samples) {
            res_sum += gather(i_r, sample_idx);
        }
    }

    return res_sum;
}

#endif //KIRCHHOFF_MIGRATION_BY_POINTS_IMPL_H_
