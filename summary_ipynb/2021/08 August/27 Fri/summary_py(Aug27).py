# sample from multiprocessing
import sys
import os
from copy import deepcopy
import argparse
from qiskit.algorithms.optimizers import SPSA
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from tqdm import tqdm

sys.path.extend(['/home/quic/QUIC-Projects/'])

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--repeat", type=int, default=10, help="number of experiment to repeat")
parser.add_argument("-I", "--log-iteration", type=int, default=10, help="log2 of number of iterations for SPSA")
parser.add_argument("-A", "--last-avg", type=int, default=1,
                    help="number of last steps to average parameters for SPSA")
parser.add_argument("--no-tracking", action="store_true",
                    help="track optimization process (optimizer.callback = 'callback')")
parser.add_argument("--tqdm", action="store_true", help="show tqdm progress bar")
args = parser.parse_args()

if __name__ == '__main__':

    from classifiers.callback import CostParamStorage
    from classifiers.quantum.qasvm import NormQSVM
    from process_manager import ProcessManager

    dim = 2
    X, y, Xt, yt = ad_hoc_data(
        training_size=2 ** 2,
        test_size=2 ** 2,
        n=dim,
        plot_data=False,
        one_hot=False,
        gap=0.3
    )

    feature_map = ZZFeatureMap(feature_dimension=dim, reps=2, entanglement='linear')
    var_form = RealAmplitudes(3, entanglement='linear', reps=3)
    backend = QuantumInstance(AerSimulator(), shots=2 ** 13, seed_simulator=None)
    nqsvm = NormQSVM(X, y, quantum_instance=backend, lamda=1, feature_map=feature_map, var_form=var_form)
    if args.no_tracking:
        optimizer = SPSA(maxiter=2 ** args.log_iteration, blocking=True, callback=None, last_avg=args.last_avg)
    else:
        storage = CostParamStorage()
        optimizer = SPSA(maxiter=2 ** args.log_iteration, blocking=True, callback=storage, last_avg=args.last_avg)


    def experiment(svm: NormQSVM, optim: SPSA, pb: tqdm = None):
        svm = deepcopy(svm)
        optim = deepcopy(optim)
        param, cost, nfev = optim.optimize(None, objective_function=svm.cost_fn, initial_point=svm.parameters)
        svm.parameters = param
        accuracy = svm.accuracy(Xt, yt)
        if pb is not None:
            pb.update(1)
        return svm, optim, accuracy

    pb = tqdm(total=args.repeat)
    manager = ProcessManager(os.cpu_count())
    result = manager(experiment, [(nqsvm, optimizer) for _ in range(args.repeat)], False, pb=pb)
    print(result)
