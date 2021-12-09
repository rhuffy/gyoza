#!/bin/bash

set -x

curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
