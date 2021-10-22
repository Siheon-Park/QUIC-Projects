import sys

from matplotlib import pyplot as plt
import numpy as np

sys.path.extend(['/home/quic/QUIC-Projects/'])

# %%

from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA

from classifiers.quantum.qasvm import NormQSVM, PseudoNormQSVM
from classifiers.callback import CostParamStorage
from classifiers.optimizer import tSPSA
from classifiers.quantum.ansatz import Circuit3, AnsatzCircuits, PQC_Properties

from multiprocessing import current_process, Pool
from slack_bot import SlackHandler, SLACK_TOKEN
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = SlackHandler(token=SLACK_TOKEN, channel='#research')
formatter = logging.Formatter(
    fmt='%(asctime)s *%(module)s* : %(message)s',
    datefmt='%H:%M:%S'
)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

BASE_DIR = Path("./tmp/Ciruit3")
BASE_DIR.mkdir(parents=True, exist_ok=True)
MAXITER = 2 ** 10
LAST_AVG = 2 ** 5
REPEATS = 2 ** 4
LAYERS = 8
DIM = 2
TRAINING_SIZE = 2 ** 4
TEST_SIZE = 2 ** 4
DATA_STRING = "At {path} : {msg}"


def run_exp(_dict: dict):
    pid = current_process()
    print(f"Running : {pid.pid}")
    reps = _dict['layer']
    exp_dir = _dict['path']
    X, y, Xt, yt, total = ad_hoc_data(
        training_size=TRAINING_SIZE,
        test_size=TEST_SIZE,
        n=DIM,
        plot_data=False,
        one_hot=False,
        gap=0.3,
        include_sample_total=True
    )
    # plt.figure(figsize=(6, 5))
    # plt.ylim(0, 2 * np.pi)
    # plt.xlim(0, 2 * np.pi)
    # plt.imshow(np.asmatrix(total).T, interpolation='nearest', origin='lower', cmap='RdBu',
    #            extent=[0, 2 * np.pi, 0, 2 * np.pi])
    #
    # plt.scatter(X[np.where(y[:] == 0), 0], X[np.where(y[:] == 0), 1],
    #             marker='s', facecolors='w', edgecolors='b', label="A train")
    # plt.scatter(X[np.where(y[:] == 1), 0], X[np.where(y[:] == 1), 1],
    #             marker='o', facecolors='w', edgecolors='r', label="B train")
    # plt.scatter(Xt[np.where(yt[:] == 0), 0], Xt[np.where(yt[:] == 0), 1],
    #             marker='s', facecolors='b', edgecolors='w', label="A test")
    # plt.scatter(Xt[np.where(yt[:] == 1), 0], Xt[np.where(yt[:] == 1), 1],
    #             marker='o', facecolors='r', edgecolors='w', label="B test")
    #
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.title("Ad hoc dataset for classification")
    #
    # plt.tight_layout()
    # plt.savefig(exp_dir / f'orinal(reps={reps})')
    # logger.info(DATA_STRING.format(path=exp_dir, msg="Dataset Generation Complete"))

    # setup
    feature_map = ZZFeatureMap(feature_dimension=DIM, reps=2, entanglement='linear')
    var_form = Circuit3(1 + int(np.log2(TRAINING_SIZE)), reps=reps)
    backend = QuantumInstance(AerSimulator(), shots=2 ** 13, seed_simulator=None)
    nqsvm = NormQSVM(X, y, quantum_instance=backend, lamda=1, feature_map=feature_map, var_form=var_form)
    # storage = CostParamStorage()
    # optimizer = SPSA(maxiter=MAXITER, blocking=True, callback=storage, last_avg=LAST_AVG)
    optimizer = SPSA(maxiter=MAXITER, blocking=True, last_avg=LAST_AVG)
    expr = PQC_Properties(var_form).expressibility()
    logger.info(DATA_STRING.format(path=exp_dir, msg=f"Ready to optimize (expr={expr})"))

    # optimize and save
    params, _, _ = optimizer.optimize(None, nqsvm.cost_fn, initial_point=nqsvm.parameters)
    nqsvm.parameters = params
    nqsvm.save(exp_dir / 'nqsvm')
    # storage.save(exp_dir / 'storage')
    acc = nqsvm.accuracy(Xt, yt)
    Xtot = np.concatenate([X, Xt])
    ytot = np.concatenate([y, yt])
    np.save(exp_dir / 'Xtot.npy', Xtot)
    np.save(exp_dir / 'ytot.npy', ytot)
    logger.info(DATA_STRING.format(path=exp_dir, msg=f"Optimization Complete (acc={acc})"))

    print(f"Finished : {pid.pid}")
    return reps, expr, acc


def main():
    paths = []
    for l in range(1, LAYERS + 1):
        for r in range(REPEATS):
            _path = BASE_DIR / f"reps={str(l)}" / str(r)
            _path.mkdir(parents=True, exist_ok=True)
            _dict = {'layer': l, 'expnum': r, 'path': _path}
            paths.append(_dict)

    print(f'number of total processes : {len(paths)}')

    with Pool(os.cpu_count()) as pool:
        exp_result = pool.map_async(run_exp, paths)
        arr = np.array(exp_result.get()).T
        np.save(BASE_DIR / "result.npy", arr)
        x, y, z = arr

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].semilogx(y, z, '.')
    axes[0, 0].set_xlabel('Expr')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Circuit3')
    axes[0, 1].plot(x, z, '.')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Circuit3')
    axes[1, 0].semilogy(x, y, '.')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Expr')
    axes[1, 0].set_title('Circuit3')

    plt.tight_layout()
    plt.savefig(BASE_DIR/'result')
    plt.show()


if __name__ == '__main__':
    main()
