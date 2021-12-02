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
from datetime import datetime

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
DATA_TYPE = SETTING.get("DATA_TYPE", "Iris")
DATA_GAP = SETTING.get("DATA_GAP", 0.3)
DATA_HOT = SETTING.get("DATA_HOT", 0)
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

if DATA_TYPE not in ["Iris", "IBM_AD"]:
    raise TypeError("'DATA_TYPE' option should be either 'iris' or 'IBM_AD'.")


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
    try:
        base_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.warning(f'The experiment exists already. Delete existing files in {str(base_dir)} to fresh start.')
    with open(base_dir / 'setting.json', 'w') as g:
        json.dump(dict(
            DATETIME=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
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
            DATA_TYPE=DATA_TYPE,
            DATA_GAP=DATA_GAP,
            DATA_HOT=DATA_HOT,
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
    if DATA_TYPE == "IBM_AD":
        feature_map = ZZFeatureMap(feature_dimension=DIM)
    else:
        feature_map1 = Circuit9(DIM, reps=1)
        feature_map = feature_map1.copy()
        for _ in range(FEATURE_LAYERS - 1):
            feature_map.compose(feature_map1, inplace=True)
    sCircuit = sample_circuit(circuit_id)
    var_form = sCircuit(int(np.log2(TRAINING_SIZE)), reps=layer, )
    #pqcp = PQC_Properties(var_form)
    #expr = pqcp.expressibility()
    #entcap = pqcp.entangling_capability()
    qasvm = QSVM(
        X, y, lamda=1,
        quantum_instance=QuantumInstance(BACKEND, shots=SHOTS, seed_simulator=None),
        var_form=var_form,
        feature_map=feature_map
    )
    storage = CostParamStorage()
    optimizer = tSPSA(maxiter=MAXITER, blocking=True, last_avg=LAST_AVG, callback=storage)
    # dlogger.debug(f'NQSVM object: expr={expr}, entcap={entcap} ({stopwatch.check()}) / Start Optimization...')
    dlogger.debug(f'NQSVM object: circuit_id={circuit_id}, layer={layer} ({stopwatch.check()}) / Start Optimization...')

    for epoch in range(1, MAXITER + 1):
        optimizer.step(qasvm.cost_fn, qasvm.parameters)
        if BLOCKING:
            if storage.num_accepted() > 2 * LAST_AVG and storage.last_cost_avg(2 * LAST_AVG, ignore_rejected=True) < \
                    storage.last_cost_avg(LAST_AVG, ignore_rejected=True):
                break
            if epoch == MAXITER:
                dlogger.warning(f"Not Converged until {MAXITER}. AVG_Cost : {storage.last_cost_avg(LAST_AVG, True)}")
        if epoch % 10 == 0:
            dlogger.debug(f"Optimizing... {epoch}/{MAXITER}")
    qasvm.parameters = storage.last_avg(LAST_AVG, ignore_rejected=True)
    dlogger.debug(f"Optimizer terminated {epoch}/{MAXITER} ({stopwatch.check()})")
    last_cost = storage.last_cost_avg(LAST_AVG, ignore_rejected=True)
    dlogger.info(f"Process {pid} finished ({stopwatch.reset()})")
    qasvm.save(exp_dir / 'qasvm')
    storage.save(exp_dir / 'storage')
    return_data = {
        'dataset': dsid,
        'circuit_id': circuit_id,
        'layer': layer,
        'num_params': qasvm.num_parameters,
        'expr': -1,
        'entcap': -1,
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
        try:
            _DATA_SAVE_FLAG = False
            X = np.load(BASE_DIR / f"Dataset #{si}" / "X.npy")
            Xt = np.load(BASE_DIR / f"Dataset #{si}" / "Xt.npy")
            y = np.load(BASE_DIR / f"Dataset #{si}" / "y.npy")
            yt = np.load(BASE_DIR / f"Dataset #{si}" / "yt.npy")
        except FileNotFoundError:
            _DATA_SAVE_FLAG = True
            if DATA_TYPE == "IBM_AD":
                X, y, Xt, yt = ad_hoc_data(
                    training_size=int(TRAINING_SIZE / 2),
                    test_size=int(TEST_SIZE / 2),
                    n=DIM,
                    gap=DATA_GAP,
                    one_hot=False
                )
            else:  # Iris
                ds = IrisDataset(feature_range=(-np.pi, np.pi), true_hot=DATA_HOT)
                # ds = IrisDataset(feature_range=(-np.pi/2, np.pi/2), true_hot=DATA_HOT)
                X, y = ds.sample(TRAINING_SIZE, return_X_y=True)
                Xt = ds.data
                yt = ds.target
        for r in range(REPEATS):
            for ci in CIRCUIT_ID:
                for l in LAYERS:
                    _path = BASE_DIR / f"Dataset #{si}" / f"Circuit #{ci}" / f"layer={l}" / str(r)
                    _path.mkdir(parents=True, exist_ok=True)
                    if not (_path / 'result.json').is_file():
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
        if _DATA_SAVE_FLAG:
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
            Xt = np.load(BASE_DIR / f"Dataset #{si}" / "Xt.npy")
            yt = np.load(BASE_DIR / f"Dataset #{si}" / "yt.npy")
            logger.info(f"Dataset size: {Xt.shape}")
            for cid, l, r in product(CIRCUIT_ID, LAYERS, range(REPEATS)):
                _path = BASE_DIR / f"Dataset #{si}/Circuit #{cid}/layer={l}/{r}/"
                try:
                    with open(_path / 'qasvm', 'rb') as _nqsvm_file:
                        _nqsvm = dill.load(_nqsvm_file)
                except FileNotFoundError:
                    continue
                else:
                    exp_dicts.append({"path": _path, "qasvm": _nqsvm})

        def calculate_accuracy(_dict):
            path = _dict["path"]
            qasvm = _dict['qasvm']

            params = [(xt,) for xt in Xt]

            ret = pool.map_async(qasvm.f, params)
            logger.debug("asyn ++ 1")
            return ret, path

        returns = list(map(calculate_accuracy, exp_dicts))
        for asyn_result, path in tqdm(returns, total=len(returns)):
            result = np.array(asyn_result.get())
            if len(result.shape) != 1:  # bug fix (for using PsudoNormQSVM)
                result = result.flatten()
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

    result2 = []
    pqc_info = read_csv(f'./pqc_prop(trs={TRAINING_SIZE}).csv')
    for si, cid, ly in product(range(NUM_SETS), CIRCUIT_ID, LAYERS):
        data_df = data.loc[(data['dataset'] == si) & (data['circuit_id'] == cid) & (data['layer'] == ly)]
        min_val = min(data_df['last_cost_avg'])
        _data_df = data_df.loc[data_df['last_cost_avg'] == min_val]
        _data_df['expr'] = pqc_info.loc[(pqc_info['circuit_id']==cid) & (pqc_info['layer']==ly)]['expr'].item()
        _data_df['entcap'] = pqc_info.loc[(pqc_info['circuit_id']==cid) & (pqc_info['layer']==ly)]['entcap'].item()
        result2.append(_data_df)
    min_select_result = concat(result2, ignore_index=True)
    min_select_result.to_csv(BASE_DIR / 'summary.csv', index=False)


def get_full_results_from_json(_path):
    with open(_path / 'result.json', 'r') as f1:
        data = json.load(f1)
    with open(_path / 'full_result.json', 'r') as f2:
        data['accuracy'] = json.load(f2)['accuracy']
    return DataFrame(data=data, index=[0])


if __name__ == '__main__':
    stwatch = StopWatch()
    # try:
    # main()
    # fvector_and_acc()
    retreive_result()

    """    except Exception as e:
        client.post_message(f"{str(type(e))}: {e}", mention=True)
        raise e
    else:
        client.post_message(f"TIME CONSUMED: {stwatch.reset()}", mention=True)"""
