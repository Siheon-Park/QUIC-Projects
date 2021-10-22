import json
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.stats import entropy

sys.path.extend(['/home/quic/QUIC-Projects/'])

# from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq import IBMQ
from qiskit.algorithms.optimizers import SPSA

from classifiers.quantum.qasvm import PseudoNormQSVM
from classifiers.quantum.ansatz import sample_circuit, Circuit9
from classifiers.datasets import IrisDataset

from multiprocessing import current_process, Pool
from slack_bot import SlackHandler, SLACK_TOKEN, SlackWebClient
import logging
import os
from pathlib import Path
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

with open('./sampling_setting.json', 'r') as f:
    _setting = json.load(fp=f)

CIRCUIT_ID = _setting["CIRCUIT_ID"]
BASE_DIR = Path(_setting["BASE_DIR"])
MAXITER = _setting["MAXITER"]
LAST_AVG = _setting["LAST_AVG"]
LAYERS = _setting["LAYERS"]
FEATURE_LAYERS = _setting["FEATURE_LAYERS"]
DIM = _setting["DIM"]
TRAINING_SIZE = _setting["TRAINING_SIZE"]
SHOTS = _setting["SHOTS"]
BLOCKING = _setting["BLOCKING"]
IBM = _setting["IBM"]
if IBM:
    IBMQ.load_account()
    BACKEND = IBMQ.get_provider(hub='ibm-q', group='open', project='main') \
        .get_backend('ibmq_qasm_simulator')
else:
    BACKEND = AerSimulator()
NUM_PQC_SAMPLE = _setting["NUM_PQC_SAMPLE"]
NUM_ALPHA_SAMPLE = _setting["NUM_ALPHA_SAMPLE"]
BINS = _setting["BINS"]
BASE = _setting["BASE"]


def _make_setting(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / 'setting.json', 'w') as g:
        json.dump(dict(
            CIRCUIT_ID=CIRCUIT_ID,
            BASE_DIR=str(base_dir),
            MAXITER=MAXITER,
            LAST_AVG=LAST_AVG,
            LAYERS=LAYERS,
            FEATURE_LAYERS=FEATURE_LAYERS,
            DIM=DIM,
            TRAINING_SIZE=TRAINING_SIZE,
            SHOTS=SHOTS,
            BLOCKING=BLOCKING,
            IBM=IBM,
            BACKEND=str(BACKEND),
            NUM_PQC_SAMPLE=NUM_PQC_SAMPLE,
            NUM_ALPHA_SAMPLE=NUM_ALPHA_SAMPLE,
            BINS=BINS,
            BASE=BASE,
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


def get_optimal_statevector(_dict: dict):
    stopwatch = StopWatch()
    pid = current_process().pid
    dsid = _dict["dataset"]
    exp_dir = _dict["path"]
    X, y = _dict["training"]
    dlogger = _dict["dlogger"]
    dlogger.info(msg=f"Process {pid} start ({stopwatch.check()})")

    # setting
    feature_map1 = Circuit9(DIM, reps=1)
    feature_map = feature_map1.copy()
    for _ in range(FEATURE_LAYERS - 1):
        feature_map.compose(feature_map1, inplace=True)
    nqsvm = PseudoNormQSVM(
        X, y, lamda=1,
        quantum_instance=QuantumInstance(BACKEND, shots=SHOTS, seed_simulator=None),
        var_form=None,
        feature_map=feature_map
    )
    dlogger.debug(f'NQSVM object: with Dataset #{dsid} ({stopwatch.check()}) / Start Optimization...')
    optimizer = SPSA(maxiter=MAXITER, blocking=True, last_avg=LAST_AVG)
    params, last_cost, _ = optimizer.optimize(None, nqsvm.cost_fn, initial_point=nqsvm.parameters)
    dlogger.debug(f"Optimizer with Dataset #{dsid} terminated ({stopwatch.check()})")
    nqsvm.parameters = params
    # sv_alpha = np.sqrt(nqsvm.alpha(params)) * np.exp(1j * 2 * np.pi * np.random.rand(len(params)))
    # return_data = {
    #     'dataset': dsid,
    #     'last_cost_avg': last_cost,
    #     'statevector_real': list(sv_alpha.real),
    #     'statevector_imag': list(sv_alpha.imag)
    # }
    alpha = nqsvm.alpha(params)
    return_data = {
        'dataset': dsid,
        'last_cost_avg': last_cost,
        'alpha': list(alpha)
    }
    with open(exp_dir / 'result.json', 'w') as fp:
        json.dump(return_data, fp=fp, indent=4)
    nqsvm.save(exp_dir / 'nqsvm')
    dlogger.info(f"Process {pid} finished ({stopwatch.reset()})")
    # return sv_alpha
    return alpha

def retrive_statevector_from_json(_dict):
    exp_dir = _dict["path"]
    # with open(exp_dir / 'result.json', 'r') as fp:
    #     ret = json.load(fp=fp)
    # return np.array(ret['statevector_real']) + 1j * np.array(ret['statevector_imag'])
    with open(exp_dir / 'result.json', 'r') as fp:
        ret = json.load(fp=fp)
    return np.array(ret['alpha'])


def pqc_fidelity_dist(_dict: dict):
    stopwatch = StopWatch()
    pid = current_process().pid
    exp_dir = _dict["path"]
    pqc = _dict["pqc"]
    ci = _dict["circuit_id"]
    layer = _dict["layer"]
    dlogger = _dict["dlogger"]
    dlogger.info(msg=f"Process {pid} start ({stopwatch.check()})")
    sampled_params1 = 2 * np.pi * np.random.rand(NUM_PQC_SAMPLE, pqc.num_parameters)
    sampled_params2 = 2 * np.pi * np.random.rand(NUM_PQC_SAMPLE, pqc.num_parameters)
    pqc_fsamples = np.empty(NUM_PQC_SAMPLE)
    for i in range(NUM_PQC_SAMPLE):
        sv1 = Statevector(pqc.assign_parameters(dict(zip(pqc.parameters, sampled_params1[i]))))
        sv2 = Statevector(pqc.assign_parameters(dict(zip(pqc.parameters, sampled_params2[i]))))
        pqc_fsamples[i] = state_fidelity(sv1, sv2)
    pqc_pmf, bin_edges = np.histogram(
        pqc_fsamples, bins=BINS, weights=np.ones(NUM_PQC_SAMPLE) / NUM_PQC_SAMPLE, range=(0, 1)
    )
    alpha_pmf = np.load(BASE_DIR / 'alpha_dist.npy')
    kl_div = entropy(pqc_pmf, alpha_pmf, base=BASE)
    np.save(exp_dir / "pqc_dist.npy", pqc_pmf)
    with open(exp_dir / 'result.json', 'w') as fp:
        json.dump({"circuit_id": ci, "layer": layer, "kl_div": kl_div, "dist": list(pqc_pmf)}, fp=fp, indent=4)
    return kl_div, ci, layer


def retrive_pqc_fidelity_dist_from_json(_dict):
    exp_dir = _dict["path"]
    with open(exp_dir / 'result.json', 'r') as fp:
        ret = json.load(fp=fp)
    return ret["kl_div"], ret["circuit_id"], ret["layer"]


def make_alpha_dist(gen_dataset: bool = True):
    pid = current_process().pid
    data_path = []
    X = np.load('./result/trial5/Dataset #0/X.npy')
    y = np.load('./result/trial5/Dataset #0/y.npy')
    if gen_dataset:
        # ds = IrisDataset(feature_range=(-np.pi, np.pi), true_hot=0)
        for si in range(2 * NUM_ALPHA_SAMPLE):
            # X, y = ds.sample(TRAINING_SIZE, return_X_y=True)
            _path = BASE_DIR / f"Dataset #{si}"
            _path.mkdir(parents=True, exist_ok=True)
            # np.save(_path / "X", X)
            # np.save(_path / 'y', y)
            data_path.append({"dataset": si, "path": _path, "training": [X, y], "dlogger": DirLogger(logger, _path)})
        logger.info(f'[MAKE_ALPHA_DIST] number of total processes : {len(data_path)} at Parent Process {pid}')
        with Pool(os.cpu_count()) as pool:
            alpha_sv_samples = list(tqdm(pool.imap_unordered(get_optimal_statevector, data_path),
                                         total=len(data_path), desc="Sampling Alpha..."))
    else:
        for si in range(2 * NUM_ALPHA_SAMPLE):
            _path = BASE_DIR / f"Dataset #{si}"
            data_path.append({"path": _path})
        with Pool(os.cpu_count()) as pool:
            alpha_sv_samples = list(tqdm(pool.imap_unordered(retrive_statevector_from_json, data_path),
                                         total=len(data_path), desc="retriving Alpha..."))

    # alpha_fsamples = np.empty(NUM_ALPHA_SAMPLE)
    # for i in range(NUM_ALPHA_SAMPLE):
    #     sv1 = Statevector(alpha_sv_samples[2 * i])
    #     sv2 = Statevector(alpha_sv_samples[2 * i + 1])
    #     alpha_fsamples[i] = state_fidelity(sv1, sv2)
    # alpha_pmf, bin_edges = np.histogram(
    #     alpha_fsamples, bins=BINS, weights=np.ones(NUM_ALPHA_SAMPLE) / NUM_ALPHA_SAMPLE, range=(0, 1)
    # )
    # np.save(BASE_DIR / 'alpha_dist.npy', alpha_pmf)


# def main(gen_dist: bool = True):
#     _make_setting(BASE_DIR)
#     pid = current_process().pid
#     varform_path = []
#     if gen_dist:
#         for ci in CIRCUIT_ID:
#             for l in LAYERS:
#                 _path = BASE_DIR / f"Circuit #{ci}" / f"Layer{l}"
#                 _path.mkdir(parents=True, exist_ok=True)
#                 pqc = sample_circuit(ci)(int(np.log2(TRAINING_SIZE)), reps=l)
#                 varform_path.append({"pqc": pqc, "path": _path, "circuit_id": ci, "layer": l,
#                                      "dlogger": DirLogger(logger, _path)})
#         logger.info(f'[MAIN] number of total processes : {len(varform_path)} at Parent Process {pid}')
#         with Pool(os.cpu_count()) as pool:
#             result = list(tqdm(pool.imap_unordered(pqc_fidelity_dist, varform_path),
#                                total=len(varform_path), desc="Sampling PQC..."))
#     else:
#         for ci in CIRCUIT_ID:
#             for l in LAYERS:
#                 _path = BASE_DIR / f"Circuit #{ci}" / f"Layer{l}"
#                 varform_path.append({"path": _path})
#         with Pool(os.cpu_count()) as pool:
#             result = list(tqdm(pool.imap_unordered(retrive_pqc_fidelity_dist_from_json, varform_path),
#                                total=len(varform_path), desc="retriving PQC..."))
#
#     data = DataFrame(
#         data=np.array(result),
#         columns=['kl_div', 'circuit_id', 'layer']
#     )
#     g = sns.relplot(kind='line', data=data.loc[data['circuit_id'] < 5], x="layer", y="kl_div", style="circuit_id",
#                     markers=True, dashes=False)
#     plt.yscale('log')
#     g.tight_layout()
#     data.to_csv(BASE_DIR / 'data.csv')
#     plt.savefig(BASE_DIR / 'result.png')
#     client.post_file(BASE_DIR / 'setting.json', text='setting', mention=False, channels='#result')
#     client.post_file(BASE_DIR / 'result.png', text='result fig', mention=True, channels='#result')
    # plt.show()


if __name__ == '__main__':
    stwatch = StopWatch()
    try:
        make_alpha_dist(gen_dataset=True)
        # main(gen_dist=True)
    except Exception as e:
        logger.error(f"{str(type(e))}: {e}")
        raise e
    else:
        client.post_message(f"TIME CONSUMED: {stwatch.reset()}", mention=True)
