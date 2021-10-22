import sys
import time

from matplotlib import pyplot as plt
import numpy as np

sys.path.extend(['/home/quic/QUIC-Projects/'])

# %%

# from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA

from classifiers.quantum.qasvm import NormQSVM, PseudoNormQSVM
from classifiers.callback import CostParamStorage
from classifiers.optimizer import tSPSA
from classifiers.quantum.ansatz import sample_circuit, PQC_Properties, Circuit9
from classifiers.datasets import IrisDataset

from multiprocessing import current_process, Pool
from slack_bot import SlackHandler, SLACK_TOKEN, SlackWebClient
import logging
import os
from pathlib import Path
import dill
from tqdm import tqdm

from pandas import DataFrame
import seaborn as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
hdlr = SlackHandler(token=SLACK_TOKEN)
formatter = logging.Formatter(
    fmt='%(asctime)s *%(module)s* : %(message)s',
    datefmt='%H:%M:%S'
)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
client = SlackWebClient()

CIRCUIT_ID = 3
BASE_DIR = Path(f"./tmp/Ciruit{CIRCUIT_ID}_iris_data")
BASE_DIR.mkdir(parents=True, exist_ok=True)
MAXITER = 2 ** 10
LAST_AVG = 2 ** 4
REPEATS = 2 ** 4
LAYERS = 8
DIM = 4
TRAINING_SIZE = 2 ** 4
TEST_SIZE = 2 ** 4
SHOTS = 2 ** 15
DATA_STRING = "At {path} : {msg}"
sCircuit = sample_circuit(CIRCUIT_ID)

def run_exp(_dict: dict):
    start = time.time()
    pid = current_process()
    reps = _dict['layer']
    exp_dir = _dict['path']
    logger.info(DATA_STRING.format(path=exp_dir, msg=f"Running : {pid.pid}"))
    # X, y, Xt, yt, total = ad_hoc_data(
    #     training_size=TRAINING_SIZE,
    #     test_size=TEST_SIZE,
    #     n=DIM,
    #     plot_data=False,
    #     one_hot=False,
    #     gap=0.3,
    #     include_sample_total=True
    # )
    ds = IrisDataset(feature_range=(-np.pi, np.pi), true_hot=0)
    X, y = ds.sample(2 ** 5, return_X_y=True)
    Xt, yt = ds.sample(2 ** 5, return_X_y=True)
    logger.debug(DATA_STRING.format(path=exp_dir, msg='Data generation complete'))

    # setup
    # feature_map = ZZFeatureMap(feature_dimension=DIM, reps=2, entanglement='linear')
    feature_map = Circuit9(DIM, reps=1)
    feature_map.compose(feature_map, inplace=True)
    var_form = sCircuit(1 + int(np.log2(TRAINING_SIZE)), reps=reps)
    backend = QuantumInstance(AerSimulator(), shots=SHOTS, seed_simulator=None)
    pqcp = PQC_Properties(var_form)
    expr = pqcp.expressibility()
    entcap = pqcp.entangling_capability()
    nqsvm = NormQSVM(X, y, quantum_instance=backend, lamda=1, feature_map=feature_map, var_form=var_form)
    logger.debug(DATA_STRING.format(path=exp_dir, msg='NQSVM object generated'))
    storage = CostParamStorage()
    optimizer = tSPSA(maxiter=MAXITER, blocking=True, last_avg=LAST_AVG, callback=storage)
    logger.debug(DATA_STRING.format(path=exp_dir, msg=f"Ready to optimize (expr={expr})"))

    # optimize and save
    # params, _, _ = optimizer.optimize(None, nqsvm.cost_fn, initial_point=nqsvm.parameters)
    # nqsvm.parameters = params
    for epoch in range(MAXITER):
        optimizer.step(nqsvm.cost_fn, nqsvm.parameters)
        if storage.num_accepted() > 2 * LAST_AVG and storage.last_cost_std(2 * LAST_AVG, ignore_rejected=True) < \
                storage.last_cost_std(LAST_AVG, ignore_rejected=True):
            break
        if epoch == MAXITER - 1:
            logger.warning(DATA_STRING.format(path=exp_dir,
                                              msg=f"Not Converged until {MAXITER}. AVG_Cost : {storage.last_cost_avg(LAST_AVG, ignore_rejected=True)}"))
        if epoch % 10 == 0:
            logger.debug(DATA_STRING.format(path=exp_dir, msg=f"Optimizing... ({epoch}/{MAXITER})"))
    logger.debug(DATA_STRING.format(path=exp_dir,
                                    msg=f"Optimizer terminated after {epoch} iteration"))
    nqsvm.parameters = storage.last_avg(LAST_AVG, ignore_rejected=True)

    layers = reps
    num_params = len(nqsvm.parameters)
    num_iter = epoch
    last_cost = storage.last_cost_avg(LAST_AVG, ignore_rejected=True)
    acc = nqsvm.accuracy(Xt, yt)
    logger.debug(DATA_STRING.format(path=exp_dir, msg=f"Accuracy Calculation Complete (acc={acc})"))

    Xtot = np.concatenate([X, Xt])
    ytot = np.concatenate([y, yt])
    end = time.time()
    logger.info(DATA_STRING.format(path=exp_dir,
                                   msg=f"Finished : {pid.pid} -- time consumed : {time.strftime('%H:%M:%S', time.gmtime(end - start))}"))
    nqsvm.save(exp_dir / 'nqsvm')
    storage.save(exp_dir / 'storage')
    np.save(exp_dir / 'Xtot.npy', Xtot)
    np.save(exp_dir / 'ytot.npy', ytot)
    return layers, num_params, expr, entcap, num_iter, last_cost, acc


def main():
    pid = current_process().pid
    paths = []
    for l in range(1, LAYERS + 1):
        for r in range(REPEATS):
            _path = BASE_DIR / f"reps={str(l)}" / str(r)
            _path.mkdir(parents=True, exist_ok=True)
            _dict = {'layer': l, 'expnum': r, 'path': _path}
            paths.append(_dict)
    logger.info(f'number of total processes : {len(paths)} at Parent Process {pid}')

    with Pool(os.cpu_count()) as pool:
        # exp_result = pool.map_async(run_exp, paths)
        exp_result = list(tqdm(pool.imap_unordered(run_exp, paths), total=len(paths)))
        # arr = np.array(exp_result.get()).T
    arr = np.array(exp_result).T
    np.save(BASE_DIR / "result.npy", arr)
    data = DataFrame(data=arr.T, columns=['layers', 'num_params', 'expr', 'entcap', 'num_iter', 'last_cost', 'acc'])
    data.to_csv(BASE_DIR / 'result.csv')

    g = sns.PairGrid(data=data)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.lineplot)
    g.tight_layout()
    g.savefig(BASE_DIR / 'result.png')
    logger.info(f'Parent Process {pid} Complete!!')
    client.post_file(file_name=BASE_DIR / 'result.png',
                     text=f'var_form:{sCircuit}, feature_map:{Circuit9}',
                     mention=True)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"{type(e)}: {e}")
        raise e
