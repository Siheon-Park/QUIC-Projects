from multiprocessing import Pool, cpu_count
from itertools import product
import abc
import json

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from pandas import DataFrame, pivot_table, read_csv
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import noise, Aer
from qiskit.utils import QuantumInstance
from tqdm import tqdm

from classifiers.callback import CostParamStorage
from classifiers.datasets.dataloader import Example_4x2, Example_4xn
from classifiers.optimizer import tSPSA
from classifiers.quantum.qasvm import QASVM
import logging

_PATH = Path.home() / 'QUIC-Projects/summary_ipynb/2021/12 December/24 Fri'
EXP_PATH = _PATH / 'exp7'
EXP_PATH.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(EXP_PATH / 'logging.log')
formatter = logging.Formatter()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class SineLSfitter(object):
    def __init__(self, theta, y):
        Y = np.matrix(y).T
        sinX = np.sin(theta)
        cosX = np.cos(theta)
        oneX = np.ones_like(theta)
        X = np.matrix(np.vstack([sinX, cosX, oneX])).T

        A, B, C = np.asarray(np.linalg.inv(X.T * X) * X.T * Y).flatten()
        self._A = A
        self._B = B
        self._C = C
        self.amp = np.sqrt(A ** 2 + B ** 2)
        self.bias = C

    def __call__(self, x):
        return self._A * np.sin(x) + self._B * np.cos(x) + self._C


class DepolarizingQASVMAgent(object):
    def __init__(self, **setting):
        self.two_gates = None
        self.single_gates = None
        self.shots = None
        self.quantum_instance = None
        self.var_form = None
        self.feature_map = None
        self.noise = None
        self.noise2 = None
        self.noise_model = None
        self._setting = {}

        # defaults
        with open(_PATH / 'noisy_simulation_setting.json', 'r') as df:
            self.setting = json.load(df)
        self.setting = setting

    def depolarizing_qi(self, prob_1, prob_2=None):
        if prob_2 is None:
            prob_2 = 10 * prob_1

        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)

        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, self.single_gates)
        noise_model.add_all_qubit_quantum_error(error_2, self.two_gates)

        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        self.noise_model = noise_model

        # Perform a noise simulation
        return QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                               basis_gates=basis_gates,
                               noise_model=noise_model,
                               shots=self.shots, optimization_level=2)

    @property
    def setting(self):
        return self._setting

    @setting.setter
    def setting(self, new_setting: dict):
        self._setting.update(new_setting)
        self.noise = self.setting.get('noise', 0)
        self.noise2 = self.setting.get('noise2', None)
        self.single_gates = self.setting.get('single_gates', None)
        self.two_gates = self.setting.get('two_gates', None)
        self.quantum_instance = self.depolarizing_qi(self.noise, self.noise2)
        var_form = self.setting.get('var_form', None)
        if isinstance(var_form, list):
            self.var_form = RealAmplitudes(var_form[0], reps=var_form[1] - 1)
        else:
            self.var_form = None
        self.feature_map = self.setting.get('feature_map', None)

    def make_qasvm(self, X, y):
        k = self.setting.get('LAMBDA', 10)
        if self.feature_map is None:
            if self.var_form is None:
                qasvm = QASVM(X, y, quantum_instance=self.quantum_instance,
                              C=None, k=k, option='Bloch_uniform').dual
            else:
                qasvm = QASVM(X, y, quantum_instance=self.quantum_instance,
                              C=None, k=k, option='Bloch_sphere', var_form=self.var_form).dual
        else:
            if self.var_form is None:
                qasvm = QASVM(X, y, quantum_instance=self.quantum_instance,
                              C=None, k=k, option='uniform', feature_map=self.feature_map).dual
            else:
                qasvm = QASVM(X, y, quantum_instance=self.quantum_instance,
                              C=None, k=k, option='QASVM', feature_map=self.feature_map, var_form=self.var_form).dual
        return qasvm

    def run(self, X: np.ndarray, y: np.ndarray, Xt: np.ndarray, yt: np.ndarray):
        qasvm = self.make_qasvm(X, y)
        MAXITER = self.setting.get('MAXITER', 2 ** 10)
        LAST_AVG = self.setting.get('LAST_AVG', 16)
        storage = CostParamStorage()
        optimizer = tSPSA(maxiter=MAXITER, blocking=True, last_avg=LAST_AVG, callback=storage)

        # for _ in tqdm(range(1, MAXITER + 1)):
        for _ in range(1, MAXITER + 1):
            optimizer.step(qasvm.cost_fn, qasvm.parameters)
            # if storage.num_accepted() > 2 * LAST_AVG:
            #    s1 = storage.last_cost_avg(2 * LAST_AVG, ignore_rejected=True)
            #    s2 = storage.last_cost_avg(LAST_AVG, ignore_rejected=True)
            #    if s1 < s2:
            #        break

        qasvm.parameters = storage.last_avg(LAST_AVG, ignore_rejected=True)
        fvec = qasvm.f(Xt)
        acc = sum(np.where(fvec > 0, 1, 0) == yt) / len(yt)
        return dict(result=fvec, acc=acc, model=qasvm, record=storage)


def load_exp_results(data_type):
    DATA_DIR = 'ibmq_run_device_2'
    DEVICE = 'montreal'
    model_dir = Path.home()/f'QUIC-Projects/{DATA_DIR}/{data_type}/{DEVICE}/'

    exact_fvec = np.load(model_dir / 'exact_fvec.npy')
    real_fvec = np.load(model_dir / 'real_fvec.npy')
    sim_fvec = np.load(model_dir / 'sim_fvec.npy')
    svm_fvec = np.load(model_dir / 'svm_fvec.npy')
    uniform_fvec = np.load(model_dir / 'uniform_fvec.npy')

    X = np.load(model_dir / 'X.npy')
    y = np.load(model_dir / 'y.npy')
    Xt = np.load(model_dir / 'Xt.npy')
    yt = np.load(model_dir / 'yt.npy')
    Xtt = np.load(model_dir / 'Xtt.npy')
    ytt = np.load(model_dir / 'ytt.npy')

    # Unknown bug fix
    if len(exact_fvec) != len(yt):
        exact_fvec = exact_fvec[4:]
    if len(sim_fvec) != len(yt):
        sim_fvec = sim_fvec[4:]
    if len(real_fvec) != len(yt):
        real_fvec = real_fvec[4:]
    if len(uniform_fvec) != len(yt):
        uniform_fvec = uniform_fvec[4:]
    if len(svm_fvec) != len(ytt):
        svm_fvec = svm_fvec[4:]

    data_dict = dict(X=X, y=y, Xt=Xt, yt=yt, Xtt=Xtt, ytt=ytt)
    result_dict = dict(exact=exact_fvec, sim=sim_fvec, real=real_fvec, uniform=uniform_fvec, svm=svm_fvec)

    return data_dict, result_dict


def run_function(args):
    agent, error, logn = args
    X, y = Example_4xn(False, logn=logn)()
    data_dir, result_dir = load_exp_results('unbalanced')
    Xt = data_dir['Xt']
    yt = data_dir['yt']
    agent.setting = dict(noise=error, var_form=[logn, logn])
    result = agent.run(X, y, Xt, yt)
    logger.info(
        "p={error}, size={size} : iteration={iteration}, acc={accuracy}, params={params}".format(
            error=error, size=int(2 ** logn), iteration=result['record'].num_accepted() + 1,
            accuracy=result['acc'], params=list(result['model'].parameters)
        )
    )
    real_fvec = result['result']
    fitfunc = SineLSfitter(Xt[:, 0], real_fvec)
    return fitfunc.amp, fitfunc.bias, real_fvec


def len_function(args):
    agent, error, logn = args
    X, y = Example_4xn(False, logn=logn)()
    agent.setting = dict(noise=error, var_form=[logn, logn])
    qasvm = agent.make_qasvm(X, y)
    return qasvm.second_order_circuit.depth(), qasvm.first_order_circuit.depth()


def main():
    logger.info("")
    with open(_PATH / 'noisy_simulation_setting.json', 'r') as f:
        with open(EXP_PATH / 'setting.json', 'w') as g:
            json.dump(json.load(f), g)
    logns = np.array([2, 3, 4, 5, 6])
    errors = 10 ** np.linspace(-5, -2, 8)
    # with Pool(cpu_count()) as pool:
    #     for amp, bias in tqdm(pool.imap(run_function, args_list), total=len(args_list)):
    result_list = []
    from dataclasses import make_dataclass
    Data = make_dataclass("Data", [('Size', int), ('Error', float), ('Amp', float), ('Bias', float)])
    (EXP_PATH / 'fvec').mkdir(parents=True, exist_ok=True)
    for trial, logn, error in tqdm(list(product(range(4), logns, errors))):
        amp, bias, fvec = run_function((DepolarizingQASVMAgent(), error, logn))
        result_list.append(Data(int(2 ** logn), error, amp, bias))
        np.save(EXP_PATH / f'fvec/size_{int(2 ** logn)}_error_{error}_{trial}.npy', fvec)
    data = DataFrame(result_list)
    table_amp = pivot_table(data, values='Amp', index='Error', columns='Size', aggfunc=['mean', 'std'])
    table_bias = pivot_table(data, values='Bias', index='Error', columns='Size', aggfunc=['mean', 'std'])
    data.to_csv(EXP_PATH / 'data.csv', index=False)
    table_amp.to_csv(EXP_PATH / 'table_amp.csv', index=False)
    table_bias.to_csv(EXP_PATH / 'table_bias.csv', index=False)

    plt.figure()
    for n in 2 ** logns:
        plt.errorbar(table_amp['mean'][n].index.to_numpy(),
                     table_amp['mean'][n].to_numpy(),
                     yerr=table_amp['std'][n].to_numpy(), label=f'SIZE={n}')
    plt.xscale('log')
    plt.xlabel('error')
    plt.ylabel('Amp')
    plt.legend()
    plt.savefig(EXP_PATH / 'AmpError')
    plt.show()

    plt.figure()
    for n in 2 ** logns:
        plt.errorbar(table_bias['mean'][n].index.to_numpy(),
                     np.abs(table_bias['mean'][n].to_numpy()),
                     yerr=table_bias['std'][n].to_numpy(), label=f'SIZE={n}')
    plt.xscale('log')
    plt.xlabel('error')
    plt.ylabel('Abs Bias')
    plt.legend()
    plt.savefig(EXP_PATH / 'BiasError')
    plt.show()

    plt.savefig(f'/home/quic/Desktop/noise_analysis/summary.png')


def main2():
    pass


if __name__ == '__main__':
    main()
