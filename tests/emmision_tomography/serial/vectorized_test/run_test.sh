#!/bin/bash

if [ ! -d "./build" ]; then
	mkdir build
fi

if [ ! -d "./results" ]; then
	mkdir results
fi
cd build
rm -r ./*

SIMD_EXTENSIONS=("AVX2" "SSE2" "NO_VECT")

for SIMD in ${SIMD_EXTENSIONS[*]}; do
	echo $SIMD
	rm -r ./*
	cmake ../ -"D${SIMD}=1"
	make
	mv ../optimization_report.txt .."/results/omptimization_report_${SIMD}.txt"
	./no_vect_run_test
	./vect_run_test
done
