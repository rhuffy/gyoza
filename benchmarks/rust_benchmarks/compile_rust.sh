
#!/bin/bash

(cd rust_benchmarks/mandelbrot && cargo build)
cp ./rust_benchmarks/mandelbrot/target/debug/mandelbrot ./rust_benchmarks/mandelbrot