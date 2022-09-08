import sys
from pathlib import Path
import json

import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.providers.aer import StatevectorSimulator
from qiskit.circuit.library import ZZFeatureMap
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from pandas import DataFrame

sys.path.extend(['/home/quic/QUIC-Projects/'])
from classifiers.quantum.qml.qasvm import PseudoTensorSoftQASVM
from classifiers.datasets.sklearn import SklearnDataset
from classifiers.quantum.ansatz import MultilayerCircuit9FeatureMap
from classifiers.convex.svm import CvxSoftQASVM


def load_train_and_test_data(dataset, train_size:float=2**13, test_size:float=2**7, random_state:int=None):
    data = dataset.data.to_numpy()
    label = dataset.target.to_numpy().astype(float)
    mask = (label==0) + (label==1)
    data = data[mask]
    label = label[mask]
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=train_size, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test

def reduce_and_normalize_data(n_components, X_train, X_test):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    X_train = scaler.fit_transform(X_train)
    X_train = pca.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)
    for i, d in enumerate(X_train):
        X_train[i] = d/np.linalg.norm(d)
    for i, d in enumerate(X_test):
        X_test[i] = d/np.linalg.norm(d)
    return X_train, X_test

def construct_training_and_test_quantum_kernel_matrix(feature_map:QuantumCircuit, X_train:np.ndarray, X_test:np.ndarray):
    quantum_instance = QuantumInstance(backend = StatevectorSimulator())
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance, enforce_psd=False)
    training_kernel = quantum_kernel.evaluate(X_train, X_train)
    test_kernel = quantum_kernel.evaluate(X_train, X_test)
    return training_kernel, test_kernel

def train_model(qasvm:PseudoTensorSoftQASVM, initial_params:np.ndarray, steps:int=2**10, writer:SummaryWriter=None, stepsize:float=0.01):
    params = initial_params
    opt = qml.AdamOptimizer(stepsize=stepsize)
    cost_list = []
    for step in range(1, steps+1):
        params, cost = opt.step_and_cost(qasvm.cost_fn, params)
        cost_list.append(cost.item())
        if writer is not None:
            writer.add_scalar('Training/Cost', cost.item(), step)
    return params, np.array(cost_list)

def test_model(qasvm:PseudoTensorSoftQASVM, params:np.ndarray, test_kernel:np.ndarray, y_test):
    cost = qasvm.cost_fn(params).item()
    fvec = qasvm.f(test_kernel, params).numpy()
    esty = np.where(fvec>0, 1, 0)
    accuracy = accuracy_score(esty, y_test)
    return fvec, accuracy, cost

def train_and_test_reference(cvxsvm:CvxSoftQASVM, train_kernel:np.ndarray, test_kernel:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
    cvxsvm.fit(train_kernel, y_train)
    fvec = cvxsvm.f(test_kernel.T)
    accuarcy = cvxsvm.accuracy(test_kernel.T, y_test)
    cost = cvxsvm.dual_objective_value
    return fvec, accuarcy, cost

def epsilon(fvec, true_fvec):
    return np.sqrt(np.mean(np.abs(fvec-true_fvec)**2)).item()

def make_figure(df:DataFrame):
    fig, ax = plt.subplots()
    mappable = ax.matshow(df, cmap='binary')
    ax.set_xlabel('n_layers')
    ax.set_ylabel('n_qubits')
    fig.colorbar(mappable)
    plt.show()
    return fig

def main():
    # setting
    dir_path = Path.cwd() / 'numerical_simulation'
    dir_path.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml('mnist_784')
    print('loaded mnist data set')
    test_size = 2**3
    n_steps = 2**3
    stepsize = 0.001
    n_feature = 4
    n_qubits_list = np.arange(6, 13+1, dtype=int) # 6 7 8 9 10 11 12 13
    n_layers_list = np.arange(1, 15+1, 2, dtype=int) # 1 3 5 7 9 11 13 15
    C=10**3
    lamda=10**3
    summary_writer = SummaryWriter(log_dir=dir_path)

    epsilon_summary = DataFrame({n_qubits:dict(zip(n_layers_list, np.zeros_like(n_layers_list))) for n_qubits in n_qubits_list})
    delta_summary = DataFrame({n_qubits:dict(zip(n_layers_list, np.zeros_like(n_layers_list))) for n_qubits in n_qubits_list})

    for n_qubits in n_qubits_list:
        sub_dir_path = dir_path / f'n_qubits={n_qubits}'
        sub_dir_path.mkdir(parents=True, exist_ok=True)
        # data loading
        train_size = 2**n_qubits
        print(train_size)
        X_train, y_train, X_test, y_test = load_train_and_test_data(dataset, train_size=train_size, test_size=test_size, random_state=90348)
        X_train, X_test = reduce_and_normalize_data(n_feature, X_train, X_test)
        feature_map = ZZFeatureMap(feature_dimension=n_feature, reps=3, entanglement='linear')
        train_kernel, test_kernel = construct_training_and_test_quantum_kernel_matrix(feature_map=feature_map, X_train=X_train, X_test=X_test)
        print('kernel matrix generated')

        # set reference
        cvxsvm = CvxSoftQASVM(kernel='precomputed', C=C, lamda=lamda)
        true_fvec, true_accuarcy, true_cost = train_and_test_reference(cvxsvm, train_kernel, test_kernel, y_train, y_test)

        # save data and reference
        np.save(sub_dir_path/'X_train.npy', X_train)
        np.save(sub_dir_path/'X_test.npy', X_test)
        np.save(sub_dir_path/'y_train.npy', y_train)
        np.save(sub_dir_path/'y_test.npy', y_test)
        np.save(sub_dir_path/'train_kernel.npy', train_kernel)
        np.save(sub_dir_path/'test_kernel.npy', test_kernel)
        with open(sub_dir_path/'reference.json', 'w') as fp:
            json.dump(dict(
                accuracy=true_accuarcy, last_cost = true_cost, fvec=true_fvec
            ), fp=fp, default=list)

        # ansatz setup
        device:qml.Device = qml.device('lightning.qubit', wires=n_qubits)
        def var_form(params):
            qml.BasicEntanglerLayers(params, wires=device.wires, rotation=qml.RY)
        for n_layers in n_layers_list:
            sub_sub_dir_path = sub_dir_path / f'n_layers={n_layers}'
            sub_sub_dir_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=sub_sub_dir_path)
            # training
            parameter_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=device.num_wires)
            qasvm = PseudoTensorSoftQASVM(data=train_kernel, label=y_train, C=C, lamda=lamda, device=device, feature_map=None, var_form=var_form)
            params=qml.numpy.random.random(parameter_shape, requires_grad=True)
            opt = qml.AdamOptimizer(stepsize=stepsize)
            cost_list = []
            for step in range(1, n_steps+1):
                params, cost = opt.step_and_cost(qasvm.cost_fn, params)
                cost_list.append(cost.item())
                if writer is not None:
                    writer.add_scalar('Training/Cost', cost_list[-1], step)
                    writer.add_scalar('Training/Normal_Cost', (cost_list[-1]-true_cost)/(cost_list[0]-true_cost), step)
            # test
            cost = qasvm.cost_fn(params.numpy()).item()
            fvec = qasvm.f(test_kernel, params.numpy()).numpy()
            accuracy = accuracy_score(np.where(fvec>0, 1, 0), y_test)
            epsilon_summary[n_qubits][n_layers] = epsilon(fvec, true_fvec)
            delta_summary[n_qubits][n_layers] = cost-true_cost
            summary_writer.add_figure('Test/epsilon', make_figure(epsilon_summary))
            summary_writer.add_figure('Test/delta', make_figure(delta_summary))
            # save result
            with open(sub_sub_dir_path/'result.json', 'w') as fp:
                json.dump(dict(
                    accuracy=accuracy, last_cost = cost, fvec=fvec, cost_list=cost_list
                ), fp=fp, default=list)



if __name__=='__main__':
    main()