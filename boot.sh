#!/bin/bash

set -e

python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install torch==2.3.0 wheel setuptools
env/bin/pip install torch-sparse torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
env/bin/pip install git+https://github.com/tomfu1/nablaDFT.git@hack/manifest
env/bin/pip install scikit-learn numpy PyYAML boto3
