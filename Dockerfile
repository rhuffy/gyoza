# Simulate running on EC2
FROM amazonlinux:latest

COPY benchmarks ./

# Install gcc

RUN chmod +x ./c_benchmarks/install_gcc.sh

RUN ./c_benchmarks/install_gcc.sh

# Install Rust

RUN chmod +x ./rust_benchmarks/install_rust.sh

RUN ./rust_benchmarks/install_rust.sh

# Install python 3.7

RUN chmod +x ./python_benchmarks/install_python.sh

RUN ./python_benchmarks/install_python.sh

# Compile C programs

RUN chmod +x ./c_benchmarks/compile_c.sh

RUN ./c_benchmarks/compile_c.sh

# Compile Rust programs

RUN chmod +x ./rust_benchmarks/compile_rust.sh

RUN ./rust_benchmarks/compile_rust.sh
