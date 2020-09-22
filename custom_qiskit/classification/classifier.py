import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from cvxopt import matrix, solvers
from classification.__init__ import Classifier, Optimizer
from classification.optimizer import CVXOPT

# qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.library.standard_gates import XGate, ZGate
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit import transpile, QiskitError

# custum qiskit
from custom_qiskit.quantum_encoder import Encoder

class SVM(Classifier):
    def __init__(self, data, label, kernel='power2'):
        """
            SVM

            kernel: return n by n matrix input: matrix X, Y
            @property: kernel(setter) opt_dict, 
            super @property: data, label, is_opt, name
            public: alpha, support_vector_index, support_vector
        """
        super().__init__(data, label, 'support vector machine')
        if kernel=='power2':
            self._kernel = lambda X, Y: np.abs(X @ Y.T)**2
        else:
            self._kernel = kernel
        self.alpha = None
        self.support_vector_index = None
        self.support_vector = None


    @property
    def kernel(self):
        return self._kernel

    @property
    def opt_dict(self):
        return self._opt_dict

    @kernel.setter
    def kernel(self, func):
        if self._kernel != func:
            self._kernel = func
            self._is_opt = False
            self._opt_dict = None

    def optimize(self, opt:Optimizer=CVXOPT, **kwargs):
        opt(self, **kwargs)


    def classify(self, test:np.ndarray):
        return np.sign((self.alpha*self.label).reshape(1, -1) @ self.kernel(self.data, test)) # pylint: disable=not-callable

    def plot(self, **kwargs): 
        cmap = kwargs.get('cmap', plt.cm.coolwarm)# pylint: disable=no-member
        s = kwargs.get('s', 100)
        linewidth = kwargs.get('linewidth', 1.0)
        facecolors = kwargs.get('facecolor', 'none')
        edgecolors = kwargs.get('edgecolors', 'k')
        plt.scatter(self.data[:,0], self.data[:,1], c=self.alpha*self.label)
        plt.colorbar()
        #plt.scatter(self.support_vector[:,0], self.support_vector[:,1], s=s, linewidth=linewidth, facecolors=facecolors, edgecolors=edgecolors)
        plt.grid()
        plt.title(f'{self.name}')

class SWAPclassifier(Classifier):
    def __init__(self, data, label):
        """
            SWAP classifier

            super @property: data, label, is_opt, name
            public: alpha, support_vector_index, support_vector
        """
        super().__init__(data, label, 'SWAP classifier')
        self.alpha = None
        self.quantum_circuit = None
        self.sqrt_weight_encoding_gate = None
        
        n_data = self.data.shape[0]
        d_data = self.data.shape[1]
        index_qubit_num = int(np.ceil(np.log2(n_data)))
        data_qubit_num = int(np.ceil(np.log2(d_data)))
        qr_a = QuantumRegister(1, name='ancila')
        qr_i = QuantumRegister(index_qubit_num, name='index')
        qr_x = QuantumRegister(data_qubit_num, name='data')
        qr_y = QuantumRegister(1, name='label')
        qr_xt = QuantumRegister(data_qubit_num, name='test')
        cr = ClassicalRegister(2, name='c')

        self.qreg = [qr_a, qr_i, qr_x, qr_y, qr_xt]
        self.creg = [cr]
        qc = QuantumCircuit(*self.qreg, *self.creg, name="training & classification")
        [qc.ctrl_encode(self.data[i], i, qr_x, qr_i, name=f"Data {i}") for i in range(n_data)] # data encoding
        [qc.ctrl_x(i, qr_y, qr_i) if label[i]>0 else None for i in range(n_data)] # label encoding
        qc.barrier()
        # SWAP test
        qc.h(qr_a)
        [qc.cswap(qr_a, qr_x[i], qr_xt[i]) for i in range(data_qubit_num)]
        qc.h(qr_a)
        # measure
        qc.measure(qr_a, cr[0]) # pylint: disable=no-member
        qc.measure(qr_y, cr[1]) # pylint: disable=no-member
        self.training_circ = qc

    def optimize(self, opt:Optimizer, **kwargs):
        opt(self, **kwargs)

    def classify(self, test: np.ndarray):
        pass


