from copy import deepcopy
from dataclasses import replace
import json
import sys
import time
import datetime
from itertools import product

from matplotlib import pyplot as plt
import numpy as np

sys.path.extend(['/home/quic/QUIC-Projects/'])

from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
from qiskit.providers.ibmq import IBMQ
from qiskit.algorithms.optimizers import SPSA

from classifiers.quantum.qasvm import SoftQASVM, PseudoSoftQASVM
from classifiers.callback import CostParamStorage
from classifiers.quantum.ansatz import sample_circuit, PQC_Properties, Circuit9, MultilayerCircuit9FeatureMap, SingleQubitFeatureMap
from classifiers.datasets import SklearnDataset

from multiprocessing import current_process, Pool
import logging
import os
from pathlib import Path
import dill
from tqdm import tqdm

from pandas import DataFrame, read_csv, concat
import seaborn as sns
import random
import logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
    Set the options with './run_experiment_setting.json':
    1. Saving results
    - BASE_DIR: Path to the directory where results would be saved (No Default)

    2. Data set selection
    - DATA_TYPE: Type of data such as 'iris', 'wine', ... (Default: 'iris')
    - DATA_HOT: Target label considered to be 1 (Default: 0)
    - FEATURE_RANGE: List of min(X), max(X) (Default: [np.pi, np.pi])
    - TRAINING_SIZE: Number of training data (Default: 64)
    - TEST_SIZE: Number of test data (Default: 32)
    - NUM_SETS: Number of training data sets (Default: 1)

    3. SoftQASVM setting
    - PSEUDO: T/F option for PseudoSoftQASVM (Default: True)
    - CIRCUIT_ID: List of circuit types to be tested (No Default)
    - LAYERS: List of ansatz layers to be tested (No Default)
    - FEATURE_TYPE: Type of the quantum feature map ('ZZFeatureMap', 'MultilayerCircuit9FeatureMap'(Default)) 
    - FEATURE_LAYERS: Number of layer for quantum feature map ansatz (Default: 4)
    - C: Hyperparameter C (Default: 10000)
    - LAMBDA: Hyperparameter lambda (Default: 10000)

    4. Simulator setting
    - QASM: T/F option for using qasm_simulator instead of statevector_simulator (Default: False)
    - SHOTS: Number of shots for each circuit simulation (Default: 8192)
    - IBM: T/F option for using IBM simulator (Default: False)

    5. Optimizer setting
    - MAXITER: Maximum iterations for optimizer (Default: 1024)
    - LAST_AVG: Number of samples to compute optimized cost and parameters (Default: 16)
    - BLOCKING: SPSA Blocking T/F option (Default: True)

    6. Repetition setting
    - REPEATS: Number of repeatitions on the same circuit for average and varience. (Default: 1)
    - SEED: Random seed for sampling (Default: random number in range 0~9999)
    - MULTIPROCESSING: multiprocessing option (Default: False)
"""

with open('./run_experiment_setting.json', 'r') as f:
    SETTING:dict = json.load(fp=f)

# 1. Saving results
BASE_DIR = Path(SETTING["BASE_DIR"])

# 2. Data set selection
DATA_TYPE = SETTING.get("DATA_TYPE", 'iris')
DATA_HOT = SETTING.get("DATA_HOT", 0)
FEATURE_RANGE = SETTING.get("FEATURE_RANGE", [-np.pi, np.pi])
TRAINING_SIZE = SETTING.get("TRAINING_SIZE", 2**6)
TEST_SIZE = SETTING.get("TEST_SIZE", 2**5)
NUM_SETS = SETTING.get("NUM_SETS", 1)

# 3. SoftQASVM setting
PSEUDO = SETTING.get("PSEUDO", True)
if PSEUDO:
    QSVM = PseudoSoftQASVM
else:
    QSVM = SoftQASVM
CIRCUIT_ID = SETTING["CIRCUIT_ID"]
LAYERS = SETTING["LAYERS"]
FEATURE_TYPE = SETTING.get("FEATURE_TYPE", "MultilayerCircuit9FeatureMap")
if FEATURE_TYPE=="ZZFeatureMap":
    FEATURE_MAP = ZZFeatureMap
elif FEATURE_TYPE=="MultilayerCircuit9FeatureMap":
    FEATURE_MAP = MultilayerCircuit9FeatureMap
FEATURE_LAYERS = SETTING.get("FEATURE_LAYERS", 2)
C = SETTING.get("C", 10**4)
LAMBDA = SETTING.get("LAMBDA", 10**4)


# 4. Simulator setting
QASM = SETTING.get("QASM", False)
SHOTS = SETTING.get("SHOTS", 2**13)
IBM = SETTING.get("IBM", False)
if IBM:
    IBMQ.load_account()
    if QASM:
        BACKEND = IBMQ.get_provider(hub='ibm-q', group='open', project='main').get_backend('ibmq_qasm_simulator')
    else:
        BACKEND = IBMQ.get_provider(hub='ibm-q', group='open', project='main').get_backend('ibmq_statevector_simulator')
else:
    if QASM:
        BACKEND = QasmSimulator(shots=SHOTS)
    else:
        BACKEND = StatevectorSimulator()

# 5. Optimizer setting
MAXITER = SETTING.get("MAXITER", 1024)
LAST_AVG = SETTING.get("LAST_AVG", 16)
BLOCKING = SETTING.get("BLOCKING", True)

# 6. Repetition setting
REPEATS = SETTING.get("REPEATS", 1)
SEED = SETTING.get("SEED", random.randint(0, 10**5))
MULTIPROCESSING = SETTING.get("MULTIPROCESSING", False)

def run_exp(_dict: dict):
    exp_dir = _dict["path"]
    training_data, training_label = _dict["training"]
    layer = _dict["layer"]
    circuit_id = _dict["circuit_id"]
    dsid = _dict["dataset"]

    # setup
    feature_map = FEATURE_MAP(training_data.shape[1], FEATURE_LAYERS)
    sCircuit = sample_circuit(circuit_id)
    var_form = sCircuit(int(np.log2(TRAINING_SIZE)), reps=layer)
    pqcp = PQC_Properties(var_form)
    expr = pqcp.expressibility()
    entcap = pqcp.entangling_capability()
    qasvm = QSVM(
        training_data, training_label, lamda=LAMBDA, C=C,
        quantum_instance=QuantumInstance(BACKEND, shots=SHOTS, seed_simulator=None),
        var_form=var_form,
        feature_map=feature_map
    )
    storage = CostParamStorage()
    optimizer = SPSA(maxiter=MAXITER, blocking=BLOCKING, last_avg=LAST_AVG, callback=storage, termination_checker=storage.termination_checker(last_avg=LAST_AVG))
    result = optimizer.minimize(qasvm.cost_fn, qasvm.initial_point)
    qasvm.parameters = storage.last_avg(LAST_AVG, ignore_rejected=True)
    last_cost = storage.last_cost_avg(LAST_AVG, ignore_rejected=True)
    qasvm.save(exp_dir / 'qasvm')
    storage.save(exp_dir / 'storage')
    return_data = {
        'dataset': dsid,
        'circuit_id': circuit_id,
        'layer': layer,
        'num_params': qasvm.num_parameters,
        'expr': expr,
        'entcap': entcap,
        'num_iter': len(storage),
        'last_cost_avg': last_cost,
    }
    with open(exp_dir / 'training_result.json', 'w') as f:
        json.dump(return_data, fp=f, indent=4)
    logger.info(f"Process {current_process().pid}: {str(exp_dir)} (iteration:{len(storage)})")
    logger.debug(f"result:{return_data}")

def train():
    training_dict = []
    for si in range(NUM_SETS):
        dataloader = SklearnDataset(DATA_TYPE, feature_range=FEATURE_RANGE, true_hot=DATA_HOT)
        training_data, training_label, test_data, test_lable = dataloader.sample_training_and_test_dataset((TRAINING_SIZE, TEST_SIZE), return_X_y=True, random_state = SEED+si, replace=True)
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
                        "training": [training_data, training_label],
                    }
                    training_dict.append(_dict)
        np.save(BASE_DIR / f"Dataset #{si}" / "training_data", training_data)
        np.save(BASE_DIR / f"Dataset #{si}" / "test_data", test_data)
        np.save(BASE_DIR / f"Dataset #{si}" / "training_label", training_label)
        np.save(BASE_DIR / f"Dataset #{si}" / "test_label", test_lable)

    if MULTIPROCESSING:
        with Pool(os.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(run_exp, training_dict), total=len(training_dict), desc="training..."):
                pass
    else:
        for _ in tqdm(map(run_exp, training_dict), total=len(training_dict), desc="training..."):
            pass
    logger.info('TRAINING FINISHED')

def run_decision(_dict:dict):
    exp_dir = _dict["path"]
    qasvm = _dict["qasvm"]
    test_data = _dict["test_data"]
    test_label = _dict["test_label"]
    fvec = qasvm.f(test_data)
    acc = sum(np.where(fvec > 0, 1, 0) == test_label) / len(test_label)
    with open(exp_dir / 'test_result.json', 'w') as fp:
        json.dump(dict(f=list(fvec), accuracy=acc), fp=fp, indent=4)
    logger.info(f"Process {current_process().pid}: {str(exp_dir)} (accuracy:{acc*100}%)")

def test():
    test_dict = []
    for si in range(NUM_SETS):
        test_data = np.load(BASE_DIR / f"Dataset #{si}" / "test_data.npy")
        test_label = np.load(BASE_DIR / f"Dataset #{si}" / "test_label.npy")
        for cid, l, r in product(CIRCUIT_ID, LAYERS, range(REPEATS)):
            _path = BASE_DIR / f"Dataset #{si}/Circuit #{cid}/layer={l}/{str(r)}/"
            try:
                with open(_path / 'qasvm', 'rb') as _qasvm_file:
                    _qasvm = dill.load(_qasvm_file)
            except FileNotFoundError as e:
                logger.warning(e)
            else:
                test_dict.append({"path": _path, "qasvm": _qasvm, "test_data":test_data, "test_label":test_label})

    if MULTIPROCESSING:
        with Pool(os.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(run_decision, test_dict), total=len(test_dict), desc="classifying..."):
                pass
    else:
        for _ in tqdm(map(run_decision, test_dict), total=len(test_dict), desc="classifying..."):
           pass
    logger.info('CLASSIFICATION FINISHED')

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
    with open(_path / 'training_result.json', 'r') as f1:
        data = json.load(f1)
    with open(_path / 'test_result.json', 'r') as f2:
        data['accuracy'] = json.load(f2)['accuracy']
    return DataFrame(data=data, index=[0])

def _make_setting():
    _setting = dict()
    _setting["Saving results"] = dict(BASE_DIR=str(BASE_DIR))
    _setting["Data set selection"] = dict(
        DATA_TYPE=DATA_TYPE,
        DATA_HOT=DATA_HOT,
        FEATURE_RANGE=FEATURE_RANGE,
        TRAINING_SIZE=TRAINING_SIZE,
        TEST_SIZE=TEST_SIZE,
        NUM_SETS=NUM_SETS
    )
    _setting["SoftQASVM setting"] = dict(
        PSEUDO=PSEUDO,
        QSVM=str(QSVM),
        CIRCUIT_ID=CIRCUIT_ID,
        LAYERS=LAYERS,
        FEATURE_TYPE=FEATURE_TYPE,
        FEATURE_MAP=str(FEATURE_MAP),
        FEATURE_LAYERS=FEATURE_LAYERS,
        C=C, LAMBDA=LAMBDA
    )
    _setting["Simulator setting"] = dict(
        QASM=QASM, SHOTS=SHOTS, IBM=IBM, BACKEND=str(BACKEND)
    )
    _setting["Optimizer setting"] = dict(
        MAXITER=MAXITER,
        LAST_AVG=LAST_AVG,
        BLOCKING=BLOCKING
    )
    _setting["Repetition setting"] = dict(
        REPEATS=REPEATS, SEED=SEED, MULTIPROCESSING=MULTIPROCESSING
    )
    return _setting

if __name__ == '__main__':
    args = sys.argv[1:]
    strat_time = time.time()
    __setting = _make_setting()
    if "h" in args:
        print(DESCRIPTION)
    if "p" in args:
        print("Current Setting:")
        print(json.dumps(__setting, indent=4))
    if len(args)==0:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        with open(BASE_DIR / 'setting.json', 'w') as g:
            json.dump(__setting, g, indent=4)
        train()
        test()
        retreive_result()
    # end_time = time.time()
    # duration = str(datetime.timedelta(seconds=end_time-strat_time))
    # print(duration)
    # with open(BASE_DIR / 'timer', 'w') as g:
    #     g.write(duration)
