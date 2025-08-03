#!/usr/bin/env bash
# exit on error
set -o errexit

apt-get update
apt-get install -y build-essential libopenblas-dev liblapack-dev gfortran

pip install -r requirements.txt