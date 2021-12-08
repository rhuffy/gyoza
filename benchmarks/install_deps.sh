#!/bin/bash

set -x

yum install -y gcc
curl https://sh.rustup.rs -sSf | sh -s -- -y


gcc -O0 simple_bench.c -o simple_bench

