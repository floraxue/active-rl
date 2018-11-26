#!/usr/bin/env python

import os
from os.path import exists, join
import subprocess
import time
import logging

# we cannot import env directly.
try:
    vfgpath = os.environ['VFGDIR']
except KeyError:
    vfgpath = './vfgdata'
os.environ['TZ'] = 'America/Los_Angeles'
time.tzset()
log_file = join(vfgpath, 'log/split', 'log-{}.txt'.format('-'.join(
    time.asctime().split())))
os.makedirs(os.path.dirname(log_file), exist_ok=True)
handlers = [logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()]
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, handlers=handlers)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def checkdir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder


def sync_folders(src_dir, dest_dir):
    if ':' not in dest_dir:
        if not exists(dest_dir):
            os.makedirs(dest_dir)
    cmd = 'rsync -avz {} {}'.format(src_dir, dest_dir)
    subprocess.check_call(cmd, shell=True)
