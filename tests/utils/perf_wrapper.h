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
    PROF_EVENT_HW(STALLED_CYCLES_FRONTEND) \
    PROF_EVENT_HW(STALLED_CYCLES_BACKEND) \
    PROF_EVENT_SW(PAGE_FAULTS) \
    PROF_EVENT_CACHE(L1D, READ, MISS) \
    PROF_EVENT_CACHE(L1D, READ, ACCESS) \
    PROF_EVENT_CACHE(L1D, PREFETCH, ACCESS)
    
#include "prof.h"

#include <string>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <omp.h>

namespace Events {
	constexpr std::size_t COUNT_EVENTS = 12;
	std::string events_names[COUNT_EVENTS] = {"CPU CYCLES",
											"INSTRUCTIONS",
											"CACHE REFERENCES",
											"CACHE MISSES",
											"BRANCH INSTRUCTIONS",
											"BRANCH MISSES",
											"STALLED CYCLES FRONTEND",
											"STALLED CYCLES BACKEND",
											"PAGE FAULTS",
											"L1D READ MISSES",
											"L1D READ ACCESSES",
											"L1D PREFETCH ACCESSES"};
}

std::pair<double, std::vector<uint64_t>> perf_wrapper(const std::function<void (void)>& coh_sum) {
	std::vector<uint64_t> events_values(Events::COUNT_EVENTS, 0);

	double t1, t2;

	t1 = omp_get_wtime(); 

	PROF_START();
	coh_sum();
	PROF_DO(events_values[index] += counter);
	
	t2 = omp_get_wtime();

    return std::make_pair(t2 - t1, events_values);
}

#endif //_PERF_WRAPPER_H