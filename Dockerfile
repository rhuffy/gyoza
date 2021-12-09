# Simulate running on EC2
FROM amazonlinux:latest

COPY benchmarks ./

# Install gcc

RUN yum install -y gcc

# Install Rust

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install python 3.7

RUN yum install -y python37

# Compile C programs

RUN chmod +x ./c_benchmarks/compile_c.sh
RUN ./c_benchmarks/compile_c.sh

# Compile Rust programs

RUN chmod +x ./rust_benchmarks/compile_rust.sh
RUN ./rust_benchmarks/compile_rust.sh
