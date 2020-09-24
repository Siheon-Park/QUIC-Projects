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
from qiskit import execute

# custum qiskit
from custom_qiskit.quantum_encoder import Encoder

from qiskit import BasicAer
_BACKEND = BasicAer.get_backend('qasm_simulator')
_EPS = 1e-3

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
        cls.support_vector_index = cls.alpha>=_EPS
        cls.support_vector = cls.data[cls.support_vector_index,:]


class SWAPOPT(Optimizer):
    def __init__(self, cls:Classifier, name):
        """
            Optimize SWAP classifier

            attributes:
                jobs:list
                results:list
                self.ZZvals:list

            method:
                execute_process
        """
        super().__init__(name)
        if cls.name != 'SWAP classifier': raise NotSuitableClsOptPairError
        self.jobs = []
        self.results = []
        self.ZZvals = []

    def execute_process(self, circuit:QuantumCircuit, backend, **kwargs):
        job = execute(circuit, backend, **kwargs)
        self.jobs.append(job)
        result = job.result()
        self.results.append(result)
        result = result.get_counts()
        c00 = result.get('00', 0)
        c01 = result.get('01', 0)
        c10 = result.get('10', 0)
        c11 = result.get('11', 0)
        ZZval = (c00 + c11 - c01 - c10) / sum(result.values())
        self.ZZvals.append(ZZval)
        

class Naive(SWAPOPT):
    """ optimize 2^n variables using MCMC by default """
    def __init__(self, cls:Classifier, **kwargs):
        super().__init__(cls, 'Naive SWAP')
        backend = kwargs.get('backend', _BACKEND)
        rdm = np.random.randint(0, cls.data.shape[0]-1, kwargs.get('iter', 10))
        for i in range(1+kwargs.get('iter', 10)):
            qc = QuantumCircuit(*cls.qreg, *cls.creg, name="weight & test")
            qc.encode(cls.data[rdm[i]], cls.qreg[-1]) # one of data to test
            qc.encode(cls.alpha, cls.qreg[1]) # alpha encoding
            qc.combine(cls.training_circuit)
            self.execute_process(qc, backend, **kwargs)


        