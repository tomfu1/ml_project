#!/usr/bin/env python

import concurrent.futures
import copy
from datetime import datetime
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import tempfile

import yaml

CONFIG_DIR = os.path.join('configurations', datetime.now().isoformat(timespec='seconds'))
os.makedirs(CONFIG_DIR, exist_ok=True)

def go(args):
    configurations = parse(args.file, args.gpu)
    logging.info(f'Found {len(configurations)} configurations')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.worker_threads) as e:
        _ = e.map(worker, list(enumerate(configurations)))

def parse(fname, gpu=False):
    def parse_value(k, v):
        if v == 'null': return None
        if k in ['clip', 'leaky_relu_alpha', 'learning_rate']: return float(v)
        return int(v)

    configurations = []
    with open(fname) as f:
        for line in f:
            options = []
            for option in line.strip().split(','):
                name, values = option.split('=', 1)
                options.append([(name, parse_value(name, v)) for v in values.split('|')])
            for kvs in itertools.product(*options):
                configuration = { k: v for k, v in kvs }
                configuration['seed'] = random.randint(0, 1024)
                configurations.append(configuration)
    return configurations

def worker(job):
    i, configuration = job
    original = copy.deepcopy(configuration)

    s = yaml.safe_dump(configuration).strip()
    s = '\n  '.join(s.split('\n'))
    logging.info(f'Starting configuration {i}:\n  {s}')

    configuration['model_path'] = None
    configuration['save'] = False
    configuration['use_gpu'] = use_gpu

    now = datetime.now()

    with tempfile.NamedTemporaryFile() as tmpf:
        tmpf.write(yaml.safe_dump(configuration).encode())
        tmpf.flush()
        
        try:
            output = subprocess.run(
                ['python3', 'main.py', '-j', '-c', tmpf.name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.warn(f'failed to complete configuration {i}')
            with open(os.path.join(CONFIG_DIR, f'{i}.err'), 'w') as f:
                f.write(str(e.stderr))
        else:
            stats = json.loads(output.stdout)
            result = {
                'configuration': original,
                'statistics': stats,
            }

            results = yaml.safe_dump(result).strip()
            results = '\n  '.join(results.split('\n'))
            logging.info(f'Configuration {i} complete:\n  {results}')

            with open(os.path.join(CONFIG_DIR, f'{i}.json'), 'w') as f:
                json.dump(result, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='FILE')
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-w', '--worker-threads', default=1)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    use_gpu = args.gpu
    go(args)
