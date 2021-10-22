#!/home/quic/anaconda3/envs/qiskit29/bin/python3.9

import sys

sys.path.extend(['/home/quic/QUIC-Projects/'])
import argparse
import os
from copy import deepcopy
from typing import Callable
import dill
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import logging
from slack_bot import SLACK_TOKEN, SlackHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
hdlr = SlackHandler(token=SLACK_TOKEN, channel='#research')
formatter = logging.Formatter(
    fmt='%(asctime)s *%(module)s* : %(message)s',
    datefmt='%H:%M:%S',
)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--num", type=int, default=20, help="block number in one side")
args = parser.parse_args()

xx = np.linspace(0, 2 * np.pi, args.num)
yy = np.linspace(0, 2 * np.pi, args.num)
XX, YY = np.meshgrid(xx, yy)
xxx = XX.flatten()
yyy = YY.flatten()

if __name__ == '__main__':
    with open('./nqsvm', 'rb') as f:
        nqsvm = dill.load(f)
    Xtot = np.load('./Xtot.npy')
    ytot = np.load('./ytot.npy')


    def point_calculator(x, y, svm=nqsvm):
        svm = deepcopy(svm)
        val = svm.f([np.array([x, y])])
        return val


    def predict(x, svm=nqsvm):
        svm = deepcopy(svm)
        val = svm.predict([x])
        return val


    class Wrapper(object):
        def __init__(self, func: Callable, **kwargs):
            self._func = func
            self._kwargs = kwargs

        def __call__(self, params: tuple):
            return self._func(*params, **self._kwargs)


    # start = time.time()
    # with tqdm(total=args.num ** 2, desc='2D_draw', ascii=True) as pbar:
    #     manager = MyProcessManager(os.cpu_count())
    #     result = manager(point_calculator, [(x, y) for x, y in zip(xxx, yyy)], pbar, svm=nqsvm)
    # print(f'TIME1 : {time.time()-start}')

    with Pool(2 * os.cpu_count()) as pool:
        result = list(tqdm(pool.imap(Wrapper(predict, svm=nqsvm), list(map(lambda t: (t,), Xtot))), total=len(Xtot)))
    esty = np.array(result)

    np.save('./esty.npy', esty)
    logger.info(f'target {predict} sampling finished : accuracy={sum(esty == ytot) / len(ytot)}')

    with Pool(2 * os.cpu_count()) as pool:
        result = list(tqdm(pool.imap(Wrapper(point_calculator, svm=nqsvm), list(zip(xxx, yyy))), total=len(xxx)))
    zzz = np.array(result)
    ZZ = zzz.reshape(XX.shape)
    np.save('./XYZ.npy', np.array([XX, YY, ZZ]))
    logger.info(f'target {point_calculator} sampling finished : {args.num} x {args.num}')
