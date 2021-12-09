#!/bin/bash

set -x

yum install -y gcc
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

gcc -O0 simple_bench.c -o simple_bench

(cd mandelbrot && cargo build)
cp ./mandelbrot/target/debug/mandelbrot ./mandelbrot
gcc -O0 bench_2.c -o bench_2

