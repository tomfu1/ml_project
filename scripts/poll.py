#!/usr/bin/env python

import os
import re
import subprocess

import pandas as pd
import seaborn as sns

output = subprocess.run(
    ['python', 'main.py'],
    check=True,
    stderr=subprocess.PIPE,
    text=True,
)

data = []
for i, m in enumerate(re.finditer(r'INFO:root:Epoch \d+:\s+Average Loss:\s+(?P<loss>\d+\.\d+)', output.stderr)):
    data.append({ 'epoch': i + 1, 'loss': float(m.group('loss')) })

df = pd.DataFrame.from_records(data)
df['epoch'] = df['epoch'].astype(str)
p = sns.relplot(df, x='epoch', y='loss', kind='line')
p.savefig(os.path.join('results', 'run.jpg'))
