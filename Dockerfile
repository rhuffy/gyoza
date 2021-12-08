# Simulate running on EC2
FROM amazonlinux:latest

COPY benchmarks ./

RUN chmod +x ./install_deps.sh

RUN ./install_deps.sh
