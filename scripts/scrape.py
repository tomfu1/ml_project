#!/usr/bin/env python

import glob
import json
import os
import pandas as pd
import seaborn as sns
import yaml

os.makedirs('results', exist_ok=True)

KEYS = [
    'batch_size',
    'clip',
    'decoder_fc1_size',
    'decoder_fc2_size',
    'encoder_fc1_size',
    'latent_dim',
    'leaky_relu_alpha',
    'learning_rate',
    'num_epochs',
    'reconstruction',
    'kl_divergence',
    'minimum_loss',
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
            return defaults[x]

data = []
for fname in glob.iglob(os.path.join('configurations', '**', '*.json')):
    with open(fname) as f:
        row = json.load(f)
    data.append({ k: get(row, k) for k in KEYS })
data.sort(key=lambda x: x['loss'])

with open(os.path.join('results', 'results.csv'), 'w') as out:
    out.write('batch size,clip,decoder fc1 size,decoder fc2 size,encoder fc1 size,latent dimensions,leaky relu alpha,learning rate,num_epochs,reconstruction,kl divergence,minimum loss,loss')
    for row in data:
        out.write('\n' + ','.join(str(row[k]) for k in KEYS))

df = pd.DataFrame.from_records(data)
for k in ['decoder_fc1_size', 'decoder_fc2_size', 'encoder_fc1_size', 'latent_dim', 'leaky_relu_alpha', 'learning_rate', 'num_epochs']:
    p = sns.relplot(data=df, x=k, y='loss')
    p.savefig(os.path.join('results', f'{k}_vs_loss'))
