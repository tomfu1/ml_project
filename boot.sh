#!/bin/bash

python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install torch wheel
env/bin/pip install torch-sparse torch-scatter torch-cluster
env/bin/pip install git+https://github.com/tomfu1/nablaDFT.git@hack/manifest
env/bin/pip install scikit-learn numpy PyYAML boto3
