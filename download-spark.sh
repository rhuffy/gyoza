#!/bin/bash

set -x

wget https://dlcdn.apache.org/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
tar -xzvf spark-3.2.0-bin-hadoop3.2.tgz
rm spark-3.2.0-bin-hadoop3.2.tgz
mv spark-3.2.0-bin-hadoop3.2 spark