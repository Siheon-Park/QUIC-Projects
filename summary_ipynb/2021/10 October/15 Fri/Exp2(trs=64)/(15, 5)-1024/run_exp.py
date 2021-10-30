import sys

sys.path.extend(['/home/quic/QUIC-Projects'])

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import dill
from classifiers.optimizer import tSPSA
from classifiers.callback import CostParamStorage
from tqdm import tqdm

DATA_DIR = Path('/home/quic/QUIC-Projects/summary_ipynb/2021/10 October/15 Fri/Exp2(trs=64)')

if __name__ == '__main__':
    X = np.load(DATA_DIR / 'DataSets/X.npy')
    Xt = np.load(DATA_DIR / 'DataSets/Xt.npy')
    y = np.load(DATA_DIR / 'DataSets/y.npy')
    yt = np.load(DATA_DIR / 'DataSets/yt.npy')

    with open(DATA_DIR / 'Samples/Circuit #15/layer=5/1/nqsvm', 'rb') as f:
        qasvm = dill.load(f)

    qasvm.parameters = qasvm.initial_point

    storage = CostParamStorage()
    optimizer = tSPSA(maxiter=2 ** 15, blocking=True, last_avg=16, callback=storage)
    for epoch in tqdm(range(1, 2 ** 10 + 1)):
        optimizer.step(qasvm.cost_fn, qasvm.parameters)
    qasvm.parameters = storage.last_avg(16, ignore_rejected=True)
    last_cost = storage.last_cost_avg(16, ignore_rejected=True)
    qasvm.save(DATA_DIR / '(15, 5)-1024/qasvm')
    storage.save(DATA_DIR / '(15, 5)-1024/storage')
