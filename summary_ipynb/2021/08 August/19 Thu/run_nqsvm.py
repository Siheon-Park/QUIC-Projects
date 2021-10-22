#!/home/quic/anaconda3/envs/qiskit29/bin/python3.9

import sys

sys.path.extend(['/home/quic/QUIC-Projects/'])
from process_manager import ProcessManager
import argparse
import os
from pathlib import Path
from copy import deepcopy
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import logging
from slack_bot import SLACK_TOKEN, SlackHandler
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.datasets import ad_hoc_data
from classifiers.quantum.qasvm import NormQSVM, PseudoNormQSVM
from classifiers.optimizer import tSPSA
from qiskit.algorithms.optimizers import SPSA
from classifiers.callback import CostParamStorage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = SlackHandler(token=SLACK_TOKEN, channel='#research')
formatter = logging.Formatter(
    fmt='%(asctime)s *%(module)s* : %(message)s',
    datefmt='%H:%M:%S'
)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--log-num-data", type=int, default=5, help="log2 number of data loaded")
parser.add_argument("-G", "--gap", type=float, default=0.3, help="gap btw classes")
parser.add_argument("-R", "--repeats", type=int, default=5, help="repeat of var_form")
parser.add_argument("-I", "--log-iteration", type=int, default=10, help="log2 of number of iterations for SPSA")
parser.add_argument("-A", "--last-avg", type=int, default=16, help="number of last steps")
parser.add_argument('-P', "--pixel", type=int, default=50, help="block number in one side")
parser.add_argument("-D", "--dir", type=str, default='./figs/', help="dir to save figures")
args = parser.parse_args()

log_num_data = args.log_num_data
repeats = args.repeats
num_iteration = 2 ** args.log_iteration
last_avg = args.last_avg
lep = args.pixel
gap = args.gap


# def plot_data(X, y, Xt, yt, total, path=figdir / 'datadist.png'):
#     plt.figure(figsize=(6, 5))
#     plt.ylim(0, 2 * np.pi)
#     plt.xlim(0, 2 * np.pi)
#     plt.imshow(np.asmatrix(total).T, interpolation='nearest', origin='lower', cmap='RdBu',
#                extent=[0, 2 * np.pi, 0, 2 * np.pi])
#
#     plt.scatter(X[np.where(y[:] == 0), 0], X[np.where(y[:] == 0), 1],
#                 marker='s', facecolors='w', edgecolors='b', label="A train")
#     plt.scatter(X[np.where(y[:] == 1), 0], X[np.where(y[:] == 1), 1],
#                 marker='o', facecolors='w', edgecolors='r', label="B train")
#     plt.scatter(Xt[np.where(yt[:] == 0), 0], Xt[np.where(yt[:] == 0), 1],
#                 marker='s', facecolors='b', edgecolors='w', label="A test")
#     plt.scatter(Xt[np.where(yt[:] == 1), 0], Xt[np.where(yt[:] == 1), 1],
#                 marker='o', facecolors='r', edgecolors='w', label="B test")
#
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     plt.title("Ad hoc dataset for classification")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=200)


# def plot_svm(xpoint, ypoint, zpoint, testy, longX, longy, path=figdir / 'result.png'):
#     plt.figure(figsize=(6, 5))
#     plt.contourf(xpoint, ypoint, zpoint, levels=200, cmap='RdBu')
#     plt.colorbar()
#     plt.scatter(longX[np.where(testy == longy), 0], longX[np.where(testy == longy), 1],
#                 marker='o', facecolors='none', edgecolors='k', label="Correct")
#     plt.scatter(longX[np.where(testy != longy), 0], longX[np.where(testy != longy), 1],
#                 marker='x', facecolors='k', edgecolors='k', label="Wrong")
#     plt.title("Ad hoc dataset for classification")
#     plt.tight_layout()
#     plt.savefig(path, dpi=200)


def point_calculator(x1, x2, svm=None):
    svm = deepcopy(svm)
    val = svm.f([np.array([x1, x2])])
    return val


def predict(x, svm=None):
    svm = deepcopy(svm)
    val = svm.predict([x])
    return val


class Wrapper(object):
    def __init__(self, func: Callable, **kwargs):
        self._func = func
        self._kwargs = deepcopy(kwargs)

    def __call__(self, params: tuple):
        return self._func(*params, **self._kwargs)


def preprocessing():
    dim = 2
    X, y, Xt, yt, total = ad_hoc_data(
        training_size=2 ** (log_num_data - 1),
        test_size=2 ** (log_num_data - 1),
        n=dim,
        plot_data=False,
        one_hot=False,
        gap=gap,
        include_sample_total=True
    )

    # plot_data(X, y, Xt, yt, total)
    # logger.info('Data plotting finished')

    feature_map = ZZFeatureMap(feature_dimension=dim, reps=2, entanglement='linear')
    var_form = RealAmplitudes(log_num_data, entanglement='linear', reps=args.repeats)
    backend = QuantumInstance(AerSimulator(), shots=2 ** 13, seed_simulator=None)
    model = NormQSVM(X, y, quantum_instance=backend, lamda=1, feature_map=feature_map, var_form=var_form)
    storage = CostParamStorage()
    optimizer = tSPSA(maxiter=num_iteration, blocking=True, callback=storage, last_avg=last_avg)

    # logger.info('SPSA optimization Start')
    # for epoch in tqdm(range(num_iteration), desc='SPSA'):
    for epoch in range(num_iteration):
        optimizer.step(model.cost_fn, model.parameters)
        # if epoch % 100 == 0:
        #     logger.info(f"SPSA optimization running : {epoch}/{num_iteration}")
    model.parameters = storage.last_avg(last_avg, True)
    # model.save(figdir / 'nqsvm')
    # storage.save(figdir / 'storage')
    # logger.info('SPSA optimization finished')
    return model, np.concatenate([X, Xt]), np.concatenate([y, yt])


if __name__ == '__main__':
    main_dir_lists = []
    for repeat in range(1, repeats + 1):
        figdir = Path(args.dir) / f"RealAmplitude (N={2 ** log_num_data}, l={repeats})"
        figdir.mkdir(parents=True, exist_ok=True)
        main_dir_lists.append(figdir)


        def processing(d):
            m, Xtot, ytot = preprocessing()
            m.save(d / 'nqsvm')
            np.save(d / 'Xtot', Xtot)
            np.save(d / 'ytot', ytot)
            ret = m.accuracy(Xtot, ytot)
            logger.info(f'(N={2 ** log_num_data}, l={repeat}) | {d}: accuracy {ret}')
            return ret


        dir_list = []
        for name in list(map(str, range(50))):
            dd = figdir / name
            dd.mkdir(parents=True, exist_ok=True)
            dir_list.append(dd)

        with Pool(os.cpu_count()) as pool:
            result = list(tqdm(pool.imap_unordered(processing, dir_list), total=len(dir_list)))
        print(result)
        logger.info(f'(N={2 ** log_num_data}, l={repeat}) | {result}')
        logger.warning(f'(N={2 ** log_num_data}, l={repeat}) | result: {sum(result) / len(result)}')
    # start = time.time()
    # manager = ProcessManager(os.cpu_count())
    # result = manager(predict, [(x,) for x in Xtot], True, svm=nqsvm)['results']
    # # print(f'TIME1 : {time.time()-start}')
    # with Pool(os.cpu_count()) as pool:
    #     result = list(tqdm(pool.imap(Wrapper(predict, svm=nqsvm), list(map(lambda a: (a,), Xtot))), total=len(Xtot),
    #                        desc='Predict'))
    # esty = np.array(result)
    # np.save(figdir / 'esty.npy', esty)
    # logger.info(f'target {predict} sampling finished : accuracy={sum(esty == ytot) / len(ytot)}')
    #
    # with Pool(os.cpu_count()) as pool:
    #     result = list(
    #         tqdm(pool.imap(Wrapper(point_calculator, svm=nqsvm), list(zip(xxx, yyy))), total=len(xxx), desc='2D Draw'))
    # zzz = np.array(result)
    # ZZ = zzz.reshape(XX.shape)
    # np.save(figdir / 'XYZ.npy', np.array([XX, YY, ZZ]))
    # logger.info(f'target {point_calculator} sampling finished : {lep} x {lep}')
    #
    # # plot_svm(XX, YY, ZZ, esty, Xtot, ytot)
    # logger.info('NQASVM plotting finished')
