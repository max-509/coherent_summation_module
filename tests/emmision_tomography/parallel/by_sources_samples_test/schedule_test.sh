#!/bin/bash

if [ ! -d "./build" ]; then
	mkdir build
fi

cd build

for i in `seq 1 32`; do
	export OMP_NUM_THREADS=${i}
	cmake ../ -DAVX2=1
	make
	./vect_run_test
done
