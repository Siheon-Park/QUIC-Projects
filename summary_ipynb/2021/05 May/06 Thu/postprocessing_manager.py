#!/home/quic/anaconda3/envs/QUIC/bin/python

import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='process results.')
parser.add_argument("sampling_number", type=int, 
                    help='number of sampling')
parser.add_argument("-D", "--dir", type=str, default='models', required=False,
                    help='location to store results')
args = parser.parse_args()
for s in range(args.sampling_number):
    dir_path = Path(args.dir)/f'{s}'
    subprocess.Popen(['./postprocessing.py', '-D', str(dir_path)])



