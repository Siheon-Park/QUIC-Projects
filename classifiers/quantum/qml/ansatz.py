import sys

import numpy as np
from matplotlib import pyplot as plt
from itertools import product

from scipy.stats import rv_continuous, entropy
import pennylane as qml

class PQC_Properties(object):
    """
    special thanks to : https://github.com/String137/PQC/blob/main/PQC.py
    """
    BINS = 75
    BASE = np.e

    def __init__(self, pqc: qml.operation.Operation, param_shape:tuple, device: qml.Device):
        self.pqc = pqc
        self.param_shape = param_shape
        self.device = device
        @qml.qnode(self.device)
        def circuit(params):
            self.pqc(params)
            return qml.state()
        self.qnode = circuit
        self.fidelity = qml.qinfo.fidelity(self.qnode, self.qnode, wires0=self.device.wires, wires1=self.device.wires)

    def expressibility(self, num_samples: int = 2 ** 10):
        sampled_params1 = 2 * np.pi * np.random.rand(num_samples, *self.param_shape)
        sampled_params2 = 2 * np.pi * np.random.rand(num_samples, *self.param_shape)

        pqc_samples = np.array([self.fidelity(p1, p2) for p1, p2 in zip(sampled_params1, sampled_params2)])
        pqc_pmf, bin_edges = np.histogram(
            pqc_samples, bins=self.BINS, weights=np.ones(num_samples) / num_samples, range=(0, 1)
        )
        bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
        haar_pmf = self._haar_pdf(bin_mids, 2 ** self.device.num_wires) * np.diff(bin_edges)
        return entropy(pqc_pmf, haar_pmf, base=self.BASE)

    @staticmethod
    def _I(b, j, n, vec):
        newvec = np.zeros((2 ** (n - 1), 1), dtype=complex)
        for new_index in range(2 ** (n - 1)):
            original_index = new_index % (2 ** (n - j)) + (new_index // (2 ** (n - j))) * (2 ** (n - j + 1)) + b * (
                    2 ** (n - j))
            newvec[new_index] = vec[int(original_index)]
        return newvec

    @staticmethod
    def _D(u, v, m):
        dist = sum([(1 / 2) * np.abs(u[i] * v[j] - u[j] * v[i]) ** 2 for i, j in product(range(m), repeat=2)])
        return float(dist)

    @staticmethod
    def _Q(n, vec):
        tot = sum([
            PQC_Properties._D(PQC_Properties._I(0, j + 1, n, vec), PQC_Properties._I(1, j + 1, n, vec), 2 ** (n - 1))
            for j in range(n)
        ])
        return tot * 4 / n

    def entangling_capability(self, num_samples: int = 2 ** 10):
        sampled_params = 2 * np.pi * np.random.rand(num_samples, *self.param_shape)
        pqc_samples = np.empty(num_samples)
        for i in range(num_samples):
            sv = self.qnode(sampled_params[i])
            pqc_samples[i] = self._Q(self.device.num_wires, sv)
        tot = sum(pqc_samples)
        return tot / num_samples

    @staticmethod
    def _haar_pdf(F, N):
        return (N - 1) * ((1 - F) ** (N - 2))