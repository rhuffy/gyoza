#!/bin/bash

set -x

cd ./c_benchmarks
mkdir bin
gcc -O0 bench_1.c -o bin/bench_1
gcc -O0 bench_2.c -o bin/bench_2