import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from cvxopt import matrix, solvers
from classification.__init__ import Classifier, Optimizer, NotSuitableClsOptPairError


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

class CVXOPT(Optimizer):
    def __init__(self, cls:Classifier, **kwargs):
        super().__init__('cvx opt')
        C = 1 if kwargs.get('C')==None else kwargs.get('C')
        X = cls.data
        y = cls.label
        Y = np.array([y]).T
        Q = matrix(np.multiply(Y @ Y.T ,cls.kernel(X, X)))
        p = matrix(-np.ones(y.size))
        G = matrix(np.vstack([-np.identity(y.size), np.identity(y.size)]))
        h = matrix(np.hstack([np.zeros(y.shape), C*np.ones(y.shape)]))
        cls._opt_dict = solvers.qp(Q, p, G, h)
        cls._is_opt = True
        cls.alpha = np.ravel(cls._opt_dict['x'])
        cls.support_vector_index = cls.alpha>=np.mean(cls.alpha)
        cls.support_vector = cls.data[cls.support_vector_index,:]


class SWAPOPT(Optimizer):
    def __init__(self, cls:Classifier, name):
        super().__init__(name)
        if cls.name != 'SWAP classifier': raise NotSuitableClsOptPairError 

class Naive(SWAPOPT):
    def __init__(self, cls:Classifier, **kwargs):
        super().__init__(cls, 'Naive SWAP')
        rdm = np.random.randint(0, cls.data.shape[0]-1, kwargs.get('iter', 10))
        for i in range(kwargs.get('iter', 10)):
            qc = QuantumCircuit(*cls.qreg, *cls.creg, name="weight & test")
            qc.encode(cls.data, cls.qreg[rdm[i]])
            qc.combine(cls.training_circuit)

        