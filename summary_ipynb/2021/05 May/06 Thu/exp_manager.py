#!/home/quic/anaconda3/envs/QUIC/bin/python

import subprocess
import argparse
from pathlib import Path
import time

parser = argparse.ArgumentParser(description='sample qasvm and save results.')
parser.add_argument("sampling_number", type=int, 
                    help='number of sampling')
parser.add_argument("-D", "--dir", type=str, default='models', required=False,
                    help='location to store results')
parser.add_argument("-R", "--reps", type=int, default=1,
                    help='maximum repeatition of var_form')
parser.add_argument("-S", "--seed", type=int, default=13,
                    help='seed to generate IRIS DATA')
parser.add_argument("--last", type=int, default=16,
                    help='last average')
parser.add_argument("-N", "--lognum", type=int, default=3, 
                    help= '2^(LOGNUM) of data will be loaded')
args = parser.parse_args()
zero_start = time.time()
for s in range(args.sampling_number):
    start = time.time()
    dir_path = Path(args.dir)/f'{s}'
    plist = []
    for i in range(args.reps):
        p = subprocess.Popen(['./experiment.py', '0', '-N', str(args.lognum), '-D', str(dir_path), '--reps', f'{i+1}', '--last', str(args.last)])
        plist.append(p)

    for p in plist:
        p.wait()
    print('sampling time for {:}th: {:} m'.format(s, (time.time()-start)/60))
print('total sampling time: {:} h'.format((time.time()-zero_start)/3600))