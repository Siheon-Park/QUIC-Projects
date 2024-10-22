import json
import sys
import time
from itertools import product

from matplotlib import pyplot as plt
import numpy as np

sys.path.extend(['/home/quic/QUIC-Projects/'])

from qiskit_machine_learning.datasets import ad_hoc_data
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

from pandas import DataFrame, read_csv, concat
import seaborn as sns

with open('./run_experiment_setting.json', 'r') as f:
    SETTING = json.load(fp=f)

CIRCUIT_ID = SETTING["CIRCUIT_ID"]
BASE_DIR = Path(SETTING["BASE_DIR"])
MAXITER = SETTING["MAXITER"]
LAST_AVG = SETTING["LAST_AVG"]
REPEATS = SETTING["REPEATS"]
LAYERS = SETTING["LAYERS"]
FEATURE_LAYERS = SETTING["FEATURE_LAYERS"]
DIM = SETTING["DIM"]
TRAINING_SIZE = SETTING["TRAINING_SIZE"]
TEST_SIZE = SETTING["TEST_SIZE"]
SHOTS = SETTING["SHOTS"]
NUM_SETS = SETTING["NUM_SETS"]
BLOCKING = SETTING["BLOCKING"]
PSEUDO = SETTING["PSEUDO"]
if PSEUDO:
    QSVM = PseudoNormQSVM
else:
    QSVM = NormQSVM
IBM = SETTING["IBM"]
if IBM:
    IBMQ.load_account()
    BACKEND = IBMQ.get_provider(hub='ibm-q', group='open', project='main') \
        .get_backend('ibmq_qasm_simulator')
else:
    BACKEND = AerSimulator()


def get_logger():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(level=logging.DEBUG)
    BASE_DIR.mkdir(exist_ok=True, parents=True)
    handler = logging.FileHandler(BASE_DIR / 'logging.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s : %(message)s'))
    _logger.addHandler(handler)
    return _logger


logger = get_logger()
client = SlackWebClient()


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
    layer = _dict["layer"]
    circuit_id = _dict["circuit_id"]
    dsid = _dict["dataset"]
    dlogger = _dict["dlogger"]
    dlogger.info(msg=f"Process {pid} start ({stopwatch.check()})")

    # setup
    feature_map = ZZFeatureMap(feature_dimension=DIM)
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
    }
    with open(exp_dir / 'result.json', 'w') as f:
        json.dump(return_data, fp=f, indent=4)


def main():
    _make_setting(BASE_DIR)
    pid = current_process().pid
    paths = []
    for si in range(NUM_SETS):
        X, y, Xt, yt = ad_hoc_data(
            training_size=int(TRAINING_SIZE / 2),
            test_size=int(TEST_SIZE / 2),
            n=DIM,
            gap=0.3,
            one_hot=False
        )
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


def fvector_and_acc():
    with Pool(os.cpu_count()) as pool:
        exp_dicts = []
        for si in range(NUM_SETS):
            Xt = np.load(BASE_DIR / f"Dataset #{si}" / "Xt")
            yt = np.load(BASE_DIR / f"Dataset #{si}" / "yt")
            logger.info(f"Dataset size: {Xt.shape}")
            for cid, l, r in product(CIRCUIT_ID, LAYERS, REPEATS):
                _path = Path(['BASE_DIR']) / f"Dataset #{si}/Circuit #{cid}/layer={l}/{r}/"
                with open(_path / 'nqsvm', 'rb') as _nqsvm_file:
                    _nqsvm = dill.load(_nqsvm_file)
                exp_dicts.append({"path": _path, "nqsvm": _nqsvm})

        def calculate_accuracy(_dict):
            path = _dict["path"]
            nqsvm = _dict['nqsvm']

            params = [(xt,) for xt in Xt]

            ret = pool.map_async(nqsvm.f, params)
            logger.debug("asyn ++ 1")
            return ret, path

        returns = list(map(calculate_accuracy, exp_dicts))
        for asyn_result, path in tqdm(returns, total=len(returns)):
            result = np.array(asyn_result.get())
            acc = sum(np.where(result > 0, 1, 0) == yt) / len(yt)

            with open(path / 'full_result.json', 'w') as fp:
                json.dump(dict(f=list(result), accuracy=acc), fp=fp)
            logger.info(f"got result of {path}")

    logger.info('JOBS FINISHED')


def retreive_result():
    result = []
    for si, ci, l, r in tqdm(product(range(NUM_SETS), CIRCUIT_ID, LAYERS, range(REPEATS)), desc='retriving...'):
        _path = BASE_DIR / f"Dataset #{si}" / f"Circuit #{ci}" / f"layer={l}" / str(r)
        try:
            _result = get_full_results_from_json(_path)
        except FileNotFoundError:
            continue
        else:
            result.append(_result)
    data = concat(result, ignore_index=True)
    data.to_csv(BASE_DIR / 'sample_summary.csv', index=False)
    logger.info('sample_summary.csv')

    for aggf in ['mean', 'median', 'std', 'min', 'max']:
        temp = data.pivot_table(values=list(data.columns[4:]), index=list(data.columns[0:4]), aggfunc=aggf)
        _df = DataFrame(columns=list(temp.index.names) + list(temp.columns),
                        data=np.hstack([np.asarray(list(temp.index)), temp.to_numpy()]))
        _df.to_csv(BASE_DIR / f'summary({aggf}).csv', index=False)
    logger.info('summary(*).csv')

    result = []
    for si, cid, ly in product(range(NUM_SETS), CIRCUIT_ID, LAYERS):
        data_df = data.loc[(data['dataset'] == si) & (data['circuit_id'] == cid) & (data['layer'] == ly)]
        min_val = min(data_df['last_cost_avg'])
        result.append(data_df.loc[data_df['last_cost_avg'] == min_val])
    min_select_result = concat(result, ignore_index=True)
    min_select_result.to_csv(BASE_DIR / 'summary.csv', index=False)


def get_full_results_from_json(_path):
    with open(_path / 'result.json', 'r') as f1:
        data = json.load(f1)
    with open(_path / 'full_result.json', 'r') as f2:
        data['accuracy'] = json.load(f2)['accuracy']
    return DataFrame(data=data, index=[0])


if __name__ == '__main__':
    stwatch = StopWatch()
    try:
        main()
        fvector_and_acc()
        retreive_result()

    except Exception as e:
        client.post_message(f"{str(type(e))}: {e}", mention=True)
        raise e
    else:
        client.post_message(f"TIME CONSUMED: {stwatch.reset()}", mention=True)
