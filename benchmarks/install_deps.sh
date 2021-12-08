#!/bin/bash

set -x

yum install -y gcc

gcc simple_bench.c -o simple_bench