#ifndef _PERF_WRAPPER_H
#define _PERF_WRAPPER_H

#define PROF_USER_EVENTS_ONLY
#define PROF_EVENT_LIST \
    PROF_EVENT_HW(CPU_CYCLES) \
    PROF_EVENT_HW(INSTRUCTIONS) \
    PROF_EVENT_HW(CACHE_REFERENCES) \
    PROF_EVENT_HW(CACHE_MISSES) \
    PROF_EVENT_HW(BRANCH_INSTRUCTIONS) \
    PROF_EVENT_HW(BRANCH_MISSES) \
    PROF_EVENT_SW(PAGE_FAULTS) \
    PROF_EVENT_CACHE(L1D, READ, MISS) \
    PROF_EVENT_CACHE(L1D, READ, ACCESS) \
    PROF_EVENT_CACHE(L1D, PREFETCH, ACCESS)

#include "prof.h"

#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <functional>

namespace Events {
	constexpr std::size_t COUNT_EVENTS = 10;
	uint64_t events_values[COUNT_EVENTS];
	std::string events_names[COUNT_EVENTS] = {"CPU CYCLES",
											"INSTRUCTIONS",
											"CACHE REFERENCES",
											"CACHE MISSES",
											"BRANCH INSTRUCTIONS",
											"BRANCH MISSES",
											"PAGE FAULTS",
											"L1D READ MISSES",
											"L1D READ ACCESSES",
											"L1D PREFETCH ACCESSES"};
}

void perf_wrapper(std::function<void (void)> coh_sum, std::ofstream &measurements_file) {
	std::fill(Events::events_values, Events::events_values + Events::COUNT_EVENTS, 0);

	struct timespec t1, t2;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
	PROF_START();
	coh_sum();
	PROF_DO(Events::events_values[index] += counter);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

	double time_in_s = ((1000.0*t2.tv_sec + 1e-6*t2.tv_nsec) - (1000.0*t1.tv_sec + 1e-6*t1.tv_nsec)) / 1000.0;

	for (std::size_t i = 0; i < Events::COUNT_EVENTS; ++i) {
		measurements_file << Events::events_values[i] << ";";
	}
	measurements_file << time_in_s;
}

#endif //_PERF_WRAPPER_H