#!/usr/bin/env python

import glob
import json
import os
import sys

import pandas as pd
import seaborn as sns
import yaml

include_path = False
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    include_path = True

if not os.path.isdir('results'): os.makedirs('results')

KEYS = [
    'decoder_fc1_size',
    'decoder_fc2_size',
    'encoder_fc1_size',
    'latent_dim',
    'leaky_relu_alpha',
    'learning_rate',
    'num_epochs',
    'warmup',
    'warmup_learning_rate',
    'reconstruction',
    'kl_divergence',
    'loss',
]

with open('main.default.yaml') as f:
    defaults = yaml.safe_load(f)
def get(r, x):
    try:
        return r['statistics'][x]
    except KeyError:
        try:
            return r['configuration'][x]
        except KeyError:
            if x == 'warmup': return None
            return defaults[x]

data = []
for fname in glob.iglob(os.path.join('configurations', '**', '*.json')):
    with open(fname) as f:
        row = json.load(f)
    record = { k: get(row, k) for k in KEYS }
    record['path'] = fname
    data.append(record)
data.sort(key=lambda x: x['loss'])

with open(os.path.join('results', 'results.csv'), 'w') as out:
    header = 'decoder fc1 size,decoder fc2 size,encoder fc1 size,latent dimensions,leaky relu alpha,learning rate,num_epochs,warmup,warmup_learning_rate,reconstruction,kl divergence,loss'
    if include_path:
        header = 'path,' + header
    out.write(header)
    for row in data:
        values = ','.join(str(row[k]) for k in KEYS)
        if include_path:
            values = f'{row["path"]},{values}'
        out.write('\n' + values)

df = pd.DataFrame.from_records(data)
for k in ['decoder_fc1_size', 'decoder_fc2_size', 'encoder_fc1_size', 'latent_dim', 'leaky_relu_alpha', 'learning_rate', 'num_epochs']:
    p = sns.relplot(data=df, x=k, y='loss')
    p.savefig(os.path.join('results', f'{k}_vs_loss'))
