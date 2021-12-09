#!/bin/bash

set -x

cd rust_benchmarks
mkdir bin
(cd mandelbrot && cargo build)
cp mandelbrot/target/debug/mandelbrot bin/mandelbrot