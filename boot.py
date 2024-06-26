#!/usr/bin/env python

import os
import subprocess

if os.name == 'nt':
    pip = os.path.join('env', 'Scripts', 'pip')
else:
    pip = os.path.join('env', 'bin', 'pip')

def shell(command):
    subprocess.run(command, shell=True, check=True)

shell('python3 -m venv env')
shell(f'{pip} install --upgrade pip')
shell(f'{pip} install torch wheel setuptools')
shell(f'{pip} install torch-sparse torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html')
shell(f'{pip} install git+https://github.com/tomfu1/nablaDFT.git@hack/manifest')
shell(f'{pip} install scikit-learn numpy PyYAML boto3')
