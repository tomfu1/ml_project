#!/usr/bin/env python

from datetime import datetime, timedelta
import logging
import os
import subprocess
import sys
import time

import boto3
import yaml

HOME_DIR = os.path.expanduser('~')
SSH_ARGS = f'-o "StrictHostKeyChecking no" -i {os.path.join(HOME_DIR, ".ssh", "rj-worker.pem")} -q'

def main(config):
    instance = Instance(config)
    instance.to('boot.sh')
    instance.exec('"env TMPDIR=/home/ec2-user ./boot.sh"')
    #instance.to('main.py')
    #instance.to('main.yaml')
    #instance.to('dataset_train_2k.db')
    #instance.exec('"nohup python main.py > main.out &"', detach=True)
    #instance.go()

class Instance:
    def __init__(self, config={}):
        ec2 = boto3.client('ec2', region_name=config.get('region_name', 'us-east-2'))

        self.instance_id = ec2.run_instances(
            ImageId=config.get('image_id', 'ami-09b90e09742640522'),
            InstanceType=config.get('instance_type', 't2.micro'),
            KeyName='rj-worker',
            SecurityGroups=['rj-worker'],
            MaxCount=1,
            MinCount=1,
            UserData='''\
#!/bin/bash
yum update -y
yum install -y git g++ python3-devel
''',
        )['Instances'][0]['InstanceId']
        logging.info(f'Started instance `{self.instance_id}`. Waiting for boot sequence ...')

        def ensure_running():
            instance_json = ec2.describe_instances(
                InstanceIds=[self.instance_id],
            )['Reservations'][0]['Instances'][0]

            if instance_json['State']['Name'] != 'pending':
                assert instance_json['State']['Name'] == 'running'
                return instance_json
        try:
            instance_json = try_until(ensure_running)
        except TryUntilTimeout:
            raise RuntimeError(f'Instance `{self.instance_id}` failed to enter `running` state')
        self.public_ip = instance_json['PublicIpAddress']
        self.volume_id = instance_json['BlockDeviceMappings'][0]['Ebs']['VolumeId']

        if 'modify_volume' in config:
            assert isinstance(config['modify_volume'], int) and config['modify_volume'] > 8

            logging.info(f'Modifying volume size from 8 -> {config["modify_volume"]}')
            assert ec2.modify_volume(
                VolumeId=self.volume_id,
                Size=config['modify_volume'],
            )['ResponseMetadata']['HTTPStatusCode'] == 200

            def ensure_complete():
                state = ec2.describe_volumes_modifications(
                    VolumeIds=[self.volume_id],
                )['VolumesModifications'][0]['ModificationState']
                if state != 'modifying':
                    assert state != 'failed'
                    return True
            try:
                try_until(ensure_complete)
            except TryUntilTimeout:
                raise RuntimeError(f"Failed to modify volume `{self.volume_id}`'s size")

        logging.info('Establishing ssh connection ...')
        def ensure_user_data():
            try:
                self.exec('which git', quiet=True)
            except subprocess.CalledProcessError:
                pass
            else:
                return True
        try:
            try_until(ensure_user_data)
        except TryUntilTimeout:
            raise RuntimeError(f'Failed to establish ssh connection to instance `{self.instance_id}`')

    def exec(self, command, detach=False, **kwargs):
        shell(f'ssh {SSH_ARGS}{" -nf" if detach else ""} {self.server} {command}', **kwargs)

    def go(self):
        shell(f'ssh {SSH_ARGS} {self.server}')

    @property
    def server(self):
        return f'ec2-user@{self.public_ip}'

    def to(self, src, dst='~'):
        shell(f'scp {SSH_ARGS} {src} {self.server}:{dst}')

def shell(command, quiet=False):
    kwargs = {}
    if quiet:
        kwargs['stderr'] = subprocess.DEVNULL
        kwargs['stdout'] = subprocess.DEVNULL
    subprocess.run(command, check=True, shell=True, **kwargs)

class TryUntilTimeout(Exception):
    pass

def try_until(f, delta=timedelta(minutes=5), sleep=5):
    sentinel = datetime.now() + delta
    while datetime.now() < sentinel:
        value = f()
        if value is not None: return value
        time.sleep(sleep)
    raise TryUntilTimeout

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    try:
        with open('cloud.yaml') as f:
            config = yaml.load(f, yaml.Loader)
    except FileNotFoundError:
        config = {}
    except Exception as e:
        logging.warning(e)
        config = {}

    main(config)
