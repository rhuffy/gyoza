#!/bin/bash

set -x

cd ./c_benchmarks
gcc -O0 bench_1.c -o bench_1
gcc -O0 bench_2.c -o bench_2
