#!/bin/bash

prefix_name=emmision_tomography_parallel_by_receivers_test
vect_extension=AVX2

for i in $(seq 1 16); do
	export OMP_NUM_THREADS=${i}

	./${prefix_name}_no_vect_run_test_${vect_extension}
	./${prefix_name}_vect_run_test_${vect_extension}

done