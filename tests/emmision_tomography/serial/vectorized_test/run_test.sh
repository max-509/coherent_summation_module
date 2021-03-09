#!/bin/bash

if [ ! -d "./build" ]; then
	mkdir build
fi

if [ ! -d "./results"]; then
	mkdir results
fi

SIMD_EXTENSIONS=("AVX512", "AVX2", "SSE2", "NO_VECT")

for SIMD in $SIMD_EXTENSIONS; do
	rm -r ./*
	cmake ../ -"D${SIMD}=1"
	make
	mv ../optimization_report.txt .."/results/omptimization_report_${SIMD}.txt"
	./no_vect_run_test
	./vect_run_test
done
