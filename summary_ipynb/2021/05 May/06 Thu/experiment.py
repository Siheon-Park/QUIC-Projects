#!/home/quic/anaconda3/envs/QUIC/bin/python


import sys
quic_path = '/home/quic/QUIC-Projects'
if not quic_path in sys.path:
    sys.path.append(quic_path)

from matplotlib.pyplot import hlines
import numpy as np 
from matplotlib import pyplot as plt
# for SVM
from classifiers.convex.svm import BinarySVM
from classifiers.kernel import Kernel
from classifiers.datasets.dataloader import Example_4x2, Sklearn_DataLoader
# for QASVM
from classifiers.quantum import Qasvm_Mapping_4x2
from classifiers.quantum.quantum_circuits import AnsatzCircuit9
from classifiers.quantum.qasvm import QASVM
from classifiers.optimizer import SPSA
from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation import CompleteMeasFitter
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, NLocal, TwoLocal
from qiskit.providers.aer import AerSimulator
from qiskit import IBMQ
from qiskit.circuit.library import PauliFeatureMap
# for logging & visualization
import logging
from classifiers.callback import CostParamStorage
from classifiers.visualization import compare_svm_and_qasvm
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing
import multiprocessing.pool

parser = argparse.ArgumentParser(description='sample qasvm and save results.')
parser.add_argument("sampling_number", type=int, 
                    help='number of sampling')
parser.add_argument("-D", "--dir", type=str, default='models', required=False,
                    help='location to store results')
parser.add_argument("-R", "--reps", type=int, default=1,
                    help='repeatition of var_form')
parser.add_argument("-S", "--seed", type=int, default=13,
                    help='seed to generate IRIS DATA')
parser.add_argument("--last", type=int, default=16,
                    help='last average')
parser.add_argument("-N", "--lognum", type=int, default=3, 
                    help= '2^(LOGNUM) of data will be loaded')
args = parser.parse_args()

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def sample_qasvm(reps=1, seed=13, dir_path:Path=None, last_avg:int=16, log_num_data:int=3):
    if dir_path is None:
        dir_path = Path.cwd()/'models'
    dir_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    dl = Sklearn_DataLoader('iris', labels=(1, 2))
    X, y, _, _ = dl(2**log_num_data, true_hot=0)
    svm = BinarySVM(Kernel('Pauli', reps=1), C=None, k=10, mutation='REDUCED_QASVM')
    svm.fit(X, y)

    #var_form = RealAmplitudes(log_num_data, reps=reps)
    #var_form = TwoLocal(log_num_data, reps=reps, rotation_blocks='rz', entanglement_blocks='cx', entanglement='linear')
    var_form = AnsatzCircuit9(log_num_data, reps=reps, rotational_block='rx', entangling_block='cz')
    feature_map = PauliFeatureMap(4, reps=1)
    quantum_instance = QuantumInstance(AerSimulator(), shots=2**13)

    np.random.seed(None)
    qasvm = QASVM(X, y, 
                num_data_qubits=4, 
                var_form=var_form, 
                quantum_instance = quantum_instance, 
                feature_map = feature_map,
                C=None, k=10, option='QASVM').dual

    optimizer = SPSA(qasvm, blocking=True)
    storage = CostParamStorage(interval=1)

    epochs = 2**10

    for epoch in tqdm(range(epochs), desc=f'sampling(reps={reps})'):
        optimizer.step(storage)
        #if epoch>=last_avg and storage.data[-last_avg:]['Cost'].std()<optimizer.allowed_increase/2:
        if storage.num_accepted()>=last_avg and storage.last_cost_std(last_avg, ignore_rejected=True)<=optimizer.allowed_increase/2:
            break

    qasvm.parameters = storage.last_avg(last_avg, ignore_rejected=True)
    regression = compare_svm_and_qasvm(svm, qasvm, repeat_for_qasvm=100)
    qasvm.save(dir_path/f'qasvm(reps={reps}).pkl')
    storage.save(dir_path/f'storage(reps={reps}).pkl')
    np.save(dir_path/f'regression(reps={reps}).npy', regression)
    plt.gcf().savefig(dir_path/f'regression_fig(reps={reps}).png', dpi=100)
    g1=storage.plot_params()
    g1.fig.savefig(dir_path/f'parameters(reps={reps}).png')
    g2=storage.plot()
    g2.fig.savefig(dir_path/f'costs(reps={reps}).png')
    print(f'sampling for reps={reps} is Done')
    if reps==1:
        svm.save(dir_path/'svm.pkl')
        np.save(dir_path/'true_regression.npy', svm.f(X))



if __name__=='__main__':
    if args.sampling_number == 0:
        sample_qasvm(args.reps, args.seed, Path(args.dir), args.last, args.lognum)
    else:
        for s in range(args.sampling_number):
            dir_path = Path(args.dir)/f'{s}'
            for i in range(1, args.reps+1):
                sample_qasvm(i, args.seed, dir_path, args.last, args.lognum)