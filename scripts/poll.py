#!/usr/bin/env python

import os
import re
import subprocess

import pandas as pd
import seaborn as sns

os.makedirs('results', exist_ok=True)

if os.name == 'nt':
    args = ['env/Scripts/python.exe main.py -v -j']
else:
    args = [['env/bin/python', 'main.py', '-v', '-j']]

output = subprocess.run(
    *args,
    check=True,
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

data = []
for i, m in enumerate(re.finditer(r'DEBUG:root:Batch \d+ - \d+:\s+Loss:\s+(?P<loss>\d+\.\d+)', output.stderr)):
    data.append({ 'batch': i + 1, 'loss': float(m.group('loss')) })

df = pd.DataFrame.from_records(data)
df['batch'] = df['batch'].astype(str)
p = sns.relplot(df, x='batch', y='loss', kind='line')
p.savefig(os.path.join('results', 'run.jpg'))
with open(os.path.join('results', 'run.json'), 'w') as f:
    out.write(output.stdout)
