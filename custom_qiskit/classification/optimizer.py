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

class QpDuel(Optimizer):
    def __init__(self, cls:Classifier, **kwargs):
        '''
            solve duel quadratic programming using CVXOPT
            min_alpha alpha.T Q alpha - (ones).T alpha s.t. C>=alpha>=0, y.T*alpha=0, ((ones).T*alpha=1)
            key words:
                C: hyperparameter
                Probability:bool consider larg. parameters as probability, that is sum(alpha)=1
        '''
        super().__init__('cvx opt')
        C = kwargs.get('C', None)
        X = cls.data
        y = cls.label
        Y = np.array([y]).T
        # alpha.T Q alpha - (ones).T alpha
        Q = matrix(np.multiply(Y @ Y.T ,cls.kernel(X, X)))
        p = matrix(-np.ones(y.size))
        # C>=alpha>=0
        if C == None:
            G = matrix(-np.identity(y.size)).T
            h = matrix(np.zeros(y.shape))
        else:
            G = matrix(np.vstack([-np.identity(y.size), np.identity(y.size)]))
            h = matrix(np.hstack([np.zeros(y.shape), C*np.ones(y.shape)]))
        # y.T*alpha=0, ((ones).T*alpha=1)
        if kwargs.get('Probability') == True:
            A = matrix(np.vstack([np.ones(y.size), y]))
            b = matrix([1., 0.])
        else:
            A = matrix(y).T
            b = matrix(0.0)
        cls._opt_dict = solvers.qp(Q, p, G, h, A, b)
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

        