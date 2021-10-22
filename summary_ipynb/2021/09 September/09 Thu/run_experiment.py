import json
import sys
import time
from itertools import product

from matplotlib import pyplot as plt
import numpy as np

sys.path.extend(['/home/quic/QUIC-Projects/'])

# from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq import IBMQ
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

from pandas import DataFrame, read_csv
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

with open('./run_experiment_setting.json', 'r') as f:
    _setting = json.load(fp=f)

CIRCUIT_ID = _setting["CIRCUIT_ID"]
BASE_DIR = Path(_setting["BASE_DIR"])
MAXITER = _setting["MAXITER"]
LAST_AVG = _setting["LAST_AVG"]
REPEATS = _setting["REPEATS"]
LAYERS = _setting["LAYERS"]
FEATURE_LAYERS = _setting["FEATURE_LAYERS"]
DIM = _setting["DIM"]
TRAINING_SIZE = _setting["TRAINING_SIZE"]
TEST_SIZE = _setting["TEST_SIZE"]
SHOTS = _setting["SHOTS"]
NUM_SETS = _setting["NUM_SETS"]
BLOCKING = _setting["BLOCKING"]
PSEUDO = _setting["PSEUDO"]
if PSEUDO:
    QSVM = PseudoNormQSVM
else:
    QSVM = NormQSVM
IBM = _setting["IBM"]
if IBM:
    IBMQ.load_account()
    BACKEND = IBMQ.get_provider(hub='ibm-q', group='open', project='main') \
        .get_backend('ibmq_qasm_simulator')
else:
    BACKEND = AerSimulator()


def _make_setting(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / 'setting.json', 'w') as g:
        json.dump(dict(
            CIRCUIT_ID=CIRCUIT_ID,
            BASE_DIR=str(base_dir),
            MAXITER=MAXITER,
            LAST_AVG=LAST_AVG,
            REPEATS=REPEATS,
            LAYERS=LAYERS,
            FEATURE_LAYERS=FEATURE_LAYERS,
            DIM=DIM,
            TRAINING_SIZE=TRAINING_SIZE,
            TEST_SIZE=TEST_SIZE,
            SHOTS=SHOTS,
            NUM_SETS=NUM_SETS,
            BLOCKING=BLOCKING,
            PSEUDO=PSEUDO,
            QSVM=str(QSVM),
            IBM=IBM,
            BACKEND=str(BACKEND)
        ), g, indent=4)


class StopWatch(object):
    def __init__(self, string=True):
        self.string = string
        _time = time.time()
        self._start = _time
        self._last_check = _time

    def check(self):
        _time = time.time()
        _return = _time - self._last_check
        self._last_check = _time
        return self.pretty_time(_return) if self.string else _return

    def reset(self):
        _time = time.time()
        _return = _time - self._start
        self._start = _time
        self._last_check = _time
        return self.pretty_time(_return) if self.string else _return

    @staticmethod
    def pretty_time(_time):
        return time.strftime('%H:%M:%S', time.gmtime(_time))


class DirLogger(object):
    DATA_STRING = "Dir @ {path} | {msg}"

    def __init__(self, _logger, path):
        self.logger = _logger
        self.path = path

    def debug(self, msg):
        self.logger.debug(self.DATA_STRING.format(path=self.path, msg=msg))

    def info(self, msg):
        self.logger.info(self.DATA_STRING.format(path=self.path, msg=msg))

    def error(self, msg):
        self.logger.error(self.DATA_STRING.format(path=self.path, msg=msg))

    def warning(self, msg):
        self.logger.warning(self.DATA_STRING.format(path=self.path, msg=msg))

    def critical(self, msg):
        self.logger.critical(self.DATA_STRING.format(path=self.path, msg=msg))


def run_exp(_dict: dict):
    stopwatch = StopWatch()
    pid = current_process().pid
    exp_dir = _dict["path"]
    X, y = _dict["training"]
    Xt, yt = _dict["test"]
    layer = _dict["layer"]
    circuit_id = _dict["circuit_id"]
    dsid = _dict["dataset"]
    dlogger = _dict["dlogger"]
    dlogger.info(msg=f"Process {pid} start ({stopwatch.check()})")

    # setup
    feature_map1 = Circuit9(DIM, reps=1)
    feature_map = feature_map1.copy()
    for _ in range(FEATURE_LAYERS - 1):
        feature_map.compose(feature_map1, inplace=True)
    sCircuit = sample_circuit(circuit_id)
    var_form = sCircuit(int(np.log2(TRAINING_SIZE)), reps=layer, )
    pqcp = PQC_Properties(var_form)
    expr = pqcp.expressibility()
    entcap = pqcp.entangling_capability()
    nqsvm = QSVM(
        X, y, lamda=1,
        quantum_instance=QuantumInstance(BACKEND, shots=SHOTS, seed_simulator=None),
        var_form=var_form,
        feature_map=feature_map
    )
    storage = CostParamStorage()
    optimizer = tSPSA(maxiter=MAXITER, blocking=True, last_avg=LAST_AVG, callback=storage)
    dlogger.debug(f'NQSVM object: expr={expr}, entcap={entcap} ({stopwatch.check()}) / Start Optimization...')

    for epoch in range(1, MAXITER + 1):
        optimizer.step(nqsvm.cost_fn, nqsvm.parameters)
        if BLOCKING:
            if storage.num_accepted() > 2 * LAST_AVG and storage.last_cost_avg(2 * LAST_AVG, ignore_rejected=True) < \
                    storage.last_cost_avg(LAST_AVG, ignore_rejected=True):
                break
            if epoch == MAXITER:
                dlogger.warning(f"Not Converged until {MAXITER}. AVG_Cost : {storage.last_cost_avg(LAST_AVG, True)}")
        if epoch % 10 == 0:
            dlogger.debug(f"Optimizing... {epoch}/{MAXITER}")
    nqsvm.parameters = storage.last_avg(LAST_AVG, ignore_rejected=True)
    dlogger.debug(f"Optimizer terminated {epoch}/{MAXITER} ({stopwatch.check()})")
    accuracy = nqsvm.accuracy(Xt, yt)
    dlogger.debug(f"Accuracy: {accuracy} ({stopwatch.check()})")
    last_cost = storage.last_cost_avg(LAST_AVG, ignore_rejected=True)
    dlogger.info(f"Process {pid} finished ({stopwatch.reset()})")
    nqsvm.save(exp_dir / 'nqsvm')
    storage.save(exp_dir / 'storage')
    return_data = {
        'dataset': dsid,
        'circuit_id': circuit_id,
        'layer': layer,
        'num_params': nqsvm.num_parameters,
        'expr': expr,
        'entcap': entcap,
        'num_iter': epoch,
        'last_cost_avg': last_cost,
        'accuracy': accuracy
    }
    with open(exp_dir / 'result.json', 'w') as f:
        json.dump(return_data, fp=f, indent=4)
    return list(return_data.values())


def retrive_from_json(_dict: dict):
    exp_dir = _dict["path"]
    with open(exp_dir / 'result.json', 'r') as f:
        data = json.load(f)
    return list(data.values())


def main():
    _make_setting(BASE_DIR)
    pid = current_process().pid
    paths = []
    for si in range(NUM_SETS):
        ds = IrisDataset(feature_range=(-np.pi, np.pi), true_hot=0)
        X, y = ds.sample(TRAINING_SIZE, return_X_y=True)
        Xt, yt = ds.sample(TRAINING_SIZE, return_X_y=True)
        for r in range(REPEATS):
            for ci in CIRCUIT_ID:
                for l in LAYERS:
                    _path = BASE_DIR / f"Dataset #{si}" / f"Circuit #{ci}" / f"layer={l}" / str(r)
                    _path.mkdir(parents=True, exist_ok=True)
                    _dict = {
                        'qsvm': QSVM,
                        'dataset': si,
                        'circuit_id': ci,
                        "trial": r,
                        "path": _path,
                        "layer": l,
                        "training": [X, y],
                        "test": [Xt, yt],
                        "dlogger": DirLogger(logger, _path)
                    }
                    paths.append(_dict)
        np.save(BASE_DIR / f"Dataset #{si}" / "X", X)
        np.save(BASE_DIR / f"Dataset #{si}" / "Xt", Xt)
        np.save(BASE_DIR / f"Dataset #{si}" / "y", y)
        np.save(BASE_DIR / f"Dataset #{si}" / "yt", yt)

    logger.info(f'number of total processes : {len(paths)} at Parent Process {pid}')

    with Pool(os.cpu_count()) as pool:
        # exp_result = pool.map_async(run_exp, paths)
        # arr = np.array(result.get()).T
        for _ in tqdm(pool.imap_unordered(run_exp, paths), total=len(paths), desc="running..."):
            pass


def retreive_result():
    result = []
    for si, ci, l, r in tqdm(product(range(NUM_SETS), CIRCUIT_ID, LAYERS, range(REPEATS)), desc='retriving...'):
        _path = BASE_DIR / f"Dataset #{si}" / f"Circuit #{ci}" / f"layer={l}" / str(r)
        try:
            _result = retrive_from_json({"path": _path})
        except FileNotFoundError:
            continue
        else:
            result.append(_result)

    data = DataFrame(
        data=np.array(result),
        columns=[
            'dataset',
            'circuit_id',
            'layer',
            'num_params',
            'expr',
            'entcap',
            'num_iter',
            'last_cost_avg',
            'accuracy'
        ]
    )
    data.to_csv(BASE_DIR / 'data.csv')
    logger.info('From json Complete!!')
    return data


def pivot_and_draw(data: DataFrame):
    df = data.pivot_table(
        index=['dataset', 'circuit_id', 'layer', 'num_params', 'dataset'],
        values=['expr', 'entcap', 'accuracy', 'num_iter', 'last_cost_avg'],
        aggfunc={'expr': 'mean',
                 'entcap': 'mean',
                 'accuracy': ['mean', 'median', 'std'],
                 'num_iter': ['mean', 'median', 'std'],
                 'last_cost_avg': ['mean', 'median', 'std']}
    )
    data1 = np.array(list(df.index))
    data2 = df.to_numpy()
    if len(data1.shape) == 1:
        data1 = data1.reshape(-1, 1)
    result = DataFrame(
        data=np.hstack([data1, data2]),
        columns=list(df.index.names) + list(df.columns)
    )
    result.to_csv(BASE_DIR / 'result.csv')

    # g = sns.relplot(data=data, x='expr', y='num_iter', hue='accuracy', size='num_params', style='circuit_id')
    # plt.xscale('log')
    # g.tight_layout()
    # g.savefig(BASE_DIR / 'result.png')
    # client.post_file(
    #     file_name=BASE_DIR / 'result.png',
    #     text=f'Iris Dataset (training: {TRAINING_SIZE})',
    #     mention=True
    # )
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    ax = axes[0]
    for cid in range(1, 5):
        _data = data.loc[data['circuit_id'] == cid]
        ax.errorbar(x=_data['num_params'], y=_data[('accuracy', 'mean')], yerr=_data[('accuracy', 'std')],
                    linestyle='none', marker='', color=colors[cid], capsize=3, alpha=0.5)
        colorble = ax.scatter(x=_data['num_params'], y=_data[('accuracy', 'median')],
                              c=np.log10(_data[('expr', 'mean')]),
                              cmap='plasma', marker=f"${cid}$", s=100)
    fig.colorbar(colorble, ax=ax)
    ax.set_xlabel('num_params')
    ax.set_ylabel('accuracy')
    ax.set_title('log10(expr)')
    ax = axes[1]
    for cid in range(1, 5):
        _data = data.loc[data['circuit_id'] == cid]
        ax.errorbar(x=_data['num_params'], y=_data[('accuracy', 'mean')], yerr=_data[('accuracy', 'std')],
                    linestyle='none', marker='', color=colors[cid], capsize=3, alpha=0.5)
        colorble = ax.scatter(x=_data['num_params'], y=_data[('accuracy', 'median')], c=_data[('entcap', 'mean')],
                              cmap='plasma', marker=f"${cid}$", s=100)
    fig.colorbar(colorble, ax=ax)
    ax.set_xlabel('num_params')
    ax.set_ylabel('accuracy')
    ax.set_title('ent. cap.')
    fig.tight_layout()
    fig.savefig(BASE_DIR / 'result')
    client.post_file(BASE_DIR / 'setting.json', text='setting', mention=False, channels='#result')
    client.post_file(BASE_DIR / 'result', text='result fig', mention=True, channels='#result')
    plt.show()


if __name__ == '__main__':
    stwatch = StopWatch()
    try:
        main()
        pivot_and_draw(retreive_result())
    except Exception as e:
        logger.error(f"{str(type(e))}: {e}")
        raise e
    else:
        client.post_message(f"TIME CONSUMED: {stwatch.reset()}", mention=True)
