import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor

from CoherentSummationModule import CoherentSummation as CohSum


@nb.jit(nopython=True)
def maxmin_diff(arr):
    n = arr.size
    odd = n % 2
    if not odd:
        n -= 1
    max_val = min_val = arr[0]
    i = 1
    while i < n:
        x = arr[i]
        y = arr[i + 1]
        if x > y:
            x, y = y, x
        min_val = min(x, min_val)
        max_val = max(y, max_val)
        i += 2
    if not odd:
        x = arr[n]
        min_val = min(x, min_val)
        max_val = max(x, max_val)
    return max_val - min_val


def split_gather(filename, gather_reader, gather_remainder_idx, n_recs, old_gather_remainder):
    gather = gather_reader(filename).reshape((n_recs, -1))
    gather_for_processing, gather_remainder = np.concatenate((old_gather_remainder, gather),
                                                             axis=-1), gather[:, -gather_remainder_idx:]
    return gather_for_processing, gather_remainder


def read_gather_default(filename):
    gather = np.fromfile(filename, dtype=np.float32)
    return gather


def result_processing_default(result, idx):
    result.tofile('result_microseismic_monitoring_{}.bin'.format(idx))
    return


def emission_tomography_on_files(file_names, table_times, dt,
                                 result_processing_function=result_processing_default,
                                 read_gather_function=read_gather_default,
                                 concurrency=3,
                                 remainder_gather=None):
    assert concurrency >= 3

    n_files = len(file_names)
    assert n_files > 0

    # For optimal arguments pass array must be C-style oriented
    if np.isfortran(table_times):
        table_times = np.ascontiguousarray(table_times)

    n_points, n_recs = table_times.shape

    # Find time for split gather by processing file and remainder
    gather_remainder_time = np.max(np.array([maxmin_diff(table_times[i_p, :]) for i_p in range(n_points)]))
    gather_remainder_sample = int(gather_remainder_time / dt)

    # Can pass remainder gather from previous processing procedure
    if remainder_gather is None:
        remainder_gather = np.empty((n_recs, 0), dtype=np.float64)

    # Processing first gather file
    gather_for_processing, remainder_gather = split_gather(file_names[0],
                                                           read_gather_function,
                                                           gather_remainder_sample,
                                                           n_recs,
                                                           remainder_gather)

    coh_sum = CohSum()

    result_future = None

    with ThreadPoolExecutor(concurrency) as executor:
        for i, filename in enumerate(file_names[1:]):
            # Read gather for future procedure call
            future_gather = executor.submit(split_gather,
                                            filename,
                                            read_gather_function,
                                            gather_remainder_sample,
                                            n_recs,
                                            remainder_gather)

            # Procedure of coherent summation
            result = coh_sum.emission_tomography(gather_for_processing, table_times, dt)

            # For testing
            # result = np.zeros((len(table_times), gather_for_processing.shape[-1]), dtype=np.float64)

            # Wait previous result processing
            if result_future is not None:
                result_future.result()

            # Processing current result
            old_result = result
            print(np.max(old_result))
            result_future = executor.submit(result_processing_function, old_result, i)

            # Get current gather for coherent summation
            gather_for_processing, remainder_gather = future_gather.result()

        # Last iteration

        result = coh_sum.emission_tomography(gather_for_processing, table_times, dt)

        if result_future is not None:
            result_future.result()

        # Save result in file, example
        old_result = result
        print(np.max(old_result))
        result_future = executor.submit(result_processing_function, old_result, n_files - 1)

        result_future.result()

    return
