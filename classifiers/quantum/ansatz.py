import sys

import numpy as np
from matplotlib import pyplot as plt
from itertools import product, permutations

from qiskit.circuit import ParameterVector, Parameter
from scipy.stats import rv_continuous, entropy
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import state_fidelity, Statevector


class PQC_Properties(object):
    """
    special thanks to : https://github.com/String137/PQC/blob/main/PQC.py
    """
    BINS = 75
    BASE = np.e

    def __init__(self, pqc: QuantumCircuit):
        self.pqc = pqc

    def expressibility(self, num_samples: int = 2 ** 10):
        sampled_params1 = 2 * np.pi * np.random.rand(num_samples, self.pqc.num_parameters)
        sampled_params2 = 2 * np.pi * np.random.rand(num_samples, self.pqc.num_parameters)
        pqc_samples = np.empty(num_samples)
        for i in range(num_samples):
            sv1 = Statevector(self.pqc.assign_parameters(dict(zip(self.pqc.parameters, sampled_params1[i]))))
            sv2 = Statevector(self.pqc.assign_parameters(dict(zip(self.pqc.parameters, sampled_params2[i]))))
            pqc_samples[i] = state_fidelity(sv1, sv2)
        pqc_pmf, bin_edges = np.histogram(
            pqc_samples, bins=self.BINS, weights=np.ones(num_samples) / num_samples, range=(0, 1)
        )
        bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
        haar_pmf = self._haar_pdf(bin_mids, 2 ** self.pqc.num_qubits) * np.diff(bin_edges)
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
        sampled_params = 2 * np.pi * np.random.rand(num_samples, self.pqc.num_parameters)
        pqc_samples = np.empty(num_samples)
        for i in range(num_samples):
            _qc = self.pqc.assign_parameters(dict(zip(self.pqc.parameters, sampled_params[i])))
            sv = Statevector(_qc).data
            pqc_samples[i] = self._Q(self.pqc.num_qubits, sv)
        tot = sum(pqc_samples)
        return tot / num_samples

    @staticmethod
    def _haar_pdf(F, N):
        return (N - 1) * ((1 - F) ** (N - 2))


class Circuit1(TwoLocal):
    """
    Circuit #1
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░─
         ├──────────┤├──────────┤ ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─
         ├──────────┤├──────────┤ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─
         ├──────────┤├──────────┤ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─
         └──────────┘└──────────┘ ░
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False, skip_final_rotation_layer=True,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #1'):
        super().__init__(num_qubits, rotation_blocks=['rx', 'rz'], entanglement_blocks=None, entanglement='linear',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit2(TwoLocal):
    """
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░───■────────────
         ├──────────┤├──────────┤ ░ ┌─┴─┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ X ├──■───────
         ├──────────┤├──────────┤ ░ └───┘┌─┴─┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────┤ X ├──■──
         ├──────────┤├──────────┤ ░      └───┘┌─┴─┐
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░───────────┤ X ├
         └──────────┘└──────────┘ ░           └───┘
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False, skip_final_rotation_layer=True,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #2'):
        super().__init__(num_qubits, rotation_blocks=['rx', 'rz'], entanglement_blocks='cx', entanglement='linear',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit3(TwoLocal):
    """
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────────────────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─────■───────────────────
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rz(θ[9]) ├──────■──────
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rz(θ[10]) ├
         └──────────┘└──────────┘ ░                         └───────────┘
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False, skip_final_rotation_layer=True,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #3'):
        super().__init__(num_qubits, rotation_blocks=['rx', 'rz'], entanglement_blocks='crz', entanglement='linear',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit4(TwoLocal):
    """
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────────────────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─────■───────────────────
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rx(θ[9]) ├──────■──────
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rx(θ[10]) ├
         └──────────┘└──────────┘ ░                         └───────────┘
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False, skip_final_rotation_layer=True,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #4'):
        super().__init__(num_qubits, rotation_blocks=['rx', 'rz'], entanglement_blocks='crx', entanglement='linear',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit5(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░                                      ┌───────────┐                          »
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────■────────────■──────┤ Rz(θ[11]) ├──────────────────────────»
         ├──────────┤├──────────┤ ░ ┌────┴─────┐     │            │      └─────┬─────┘                          »
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─────┼────────────┼────────────■────────────■────────────■──────»
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐      │                   ┌─────┴─────┐      │      »
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rz(θ[9]) ├──────┼───────────────────┤ Rz(θ[12]) ├──────┼──────»
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐             └───────────┘┌─────┴─────┐»
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rz(θ[10]) ├──────────────────────────┤ Rz(θ[13]) ├»
         └──────────┘└──────────┘ ░                         └───────────┘                          └───────────┘»
    «     ┌───────────┐                          ┌───────────┐                           ░ ┌───────────┐┌───────────┐
    «q_0: ┤ Rz(θ[14]) ├──────────────────────────┤ Rz(θ[17]) ├───────────────────────────░─┤ Rx(θ[20]) ├┤ Rz(θ[24]) ├─
    «     └─────┬─────┘┌───────────┐             └─────┬─────┘┌───────────┐              ░ ├───────────┤├───────────┤
    «q_1: ──────┼──────┤ Rz(θ[15]) ├───────────────────┼──────┤ Rz(θ[18]) ├──────────────░─┤ Rx(θ[21]) ├┤ Rz(θ[25]) ├─
    «           │      └─────┬─────┘                   │      └─────┬─────┘┌───────────┐ ░ ├───────────┤├───────────┤
    «q_2: ──────■────────────■────────────■────────────┼────────────┼──────┤ Rz(θ[19]) ├─░─┤ Rx(θ[22]) ├┤ Rz(θ[26]) ├─
    «                               ┌─────┴─────┐      │            │      └─────┬─────┘ ░ ├───────────┤├───────────┤
    «q_3: ──────────────────────────┤ Rz(θ[16]) ├──────■────────────■────────────■───────░─┤ Rx(θ[23]) ├┤ Rz(θ[27]) ├─
    «                               └───────────┘                                        ░ └───────────┘└───────────┘
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #5'):
        pv_list = list(ParameterVector(parameter_prefix, (num_qubits ** 2 + 3 * num_qubits) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i, j in permutations(range(num_qubits), 2):
                self.crz(pv_list.pop(0), i, j)
            if insert_barriers:
                self.barrier()
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers and r != reps - 1:
                self.barrier()


class Circuit6(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░                                      ┌───────────┐                          »
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────■────────────■──────┤ Rx(θ[11]) ├──────────────────────────»
         ├──────────┤├──────────┤ ░ ┌────┴─────┐     │            │      └─────┬─────┘                          »
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─────┼────────────┼────────────■────────────■────────────■──────»
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐      │                   ┌─────┴─────┐      │      »
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rx(θ[9]) ├──────┼───────────────────┤ Rx(θ[12]) ├──────┼──────»
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐             └───────────┘┌─────┴─────┐»
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rx(θ[10]) ├──────────────────────────┤ Rx(θ[13]) ├»
         └──────────┘└──────────┘ ░                         └───────────┘                          └───────────┘»
    «     ┌───────────┐                          ┌───────────┐                           ░ ┌───────────┐┌───────────┐
    «q_0: ┤ Rx(θ[14]) ├──────────────────────────┤ Rx(θ[17]) ├───────────────────────────░─┤ Rx(θ[20]) ├┤ Rz(θ[24]) ├─
    «     └─────┬─────┘┌───────────┐             └─────┬─────┘┌───────────┐              ░ ├───────────┤├───────────┤
    «q_1: ──────┼──────┤ Rx(θ[15]) ├───────────────────┼──────┤ Rx(θ[18]) ├──────────────░─┤ Rx(θ[21]) ├┤ Rz(θ[25]) ├─
    «           │      └─────┬─────┘                   │      └─────┬─────┘┌───────────┐ ░ ├───────────┤├───────────┤
    «q_2: ──────■────────────■────────────■────────────┼────────────┼──────┤ Rx(θ[19]) ├─░─┤ Rx(θ[22]) ├┤ Rz(θ[26]) ├─
    «                               ┌─────┴─────┐      │            │      └─────┬─────┘ ░ ├───────────┤├───────────┤
    «q_3: ──────────────────────────┤ Rx(θ[16]) ├──────■────────────■────────────■───────░─┤ Rx(θ[23]) ├┤ Rz(θ[27]) ├─
    «                               └───────────┘                                        ░ └───────────┘└───────────┘
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #6'):
        pv_list = list(ParameterVector(parameter_prefix, (num_qubits ** 2 + 3 * num_qubits) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i, j in permutations(range(num_qubits), 2):
                self.crx(pv_list.pop(0), i, j)
            if insert_barriers:
                self.barrier()
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers and r != reps - 1:
                self.barrier()


class Circuit7(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░              ░ ┌───────────┐┌───────────┐ ░               ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────░─┤ Rx(θ[10]) ├┤ Rz(θ[14]) ├─░───────────────░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░               ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─░─┤ Rx(θ[11]) ├┤ Rz(θ[15]) ├─░───────■───────░─
         ├──────────┤├──────────┤ ░ └──────────┘ ░ ├───────────┤├───────────┤ ░ ┌─────┴─────┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■───────░─┤ Rx(θ[12]) ├┤ Rz(θ[16]) ├─░─┤ Rz(θ[18]) ├─░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░ └───────────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rz(θ[9]) ├─░─┤ Rx(θ[13]) ├┤ Rz(θ[17]) ├─░───────────────░─
         └──────────┘└──────────┘ ░ └──────────┘ ░ └───────────┘└───────────┘ ░               ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #7'):
        pv_list = list(ParameterVector(parameter_prefix, (5 * num_qubits - 1) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(2 * reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits - 1):
                if (r + i) % 2 == 0:
                    self.crz(pv_list.pop(0), i, i + 1)
            if insert_barriers and r != 2 * reps - 1:
                self.barrier()


class Circuit8(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░              ░ ┌───────────┐┌───────────┐ ░               ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────░─┤ Rx(θ[10]) ├┤ Rz(θ[14]) ├─░───────────────░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░               ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─░─┤ Rx(θ[11]) ├┤ Rz(θ[15]) ├─░───────■───────░─
         ├──────────┤├──────────┤ ░ └──────────┘ ░ ├───────────┤├───────────┤ ░ ┌─────┴─────┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■───────░─┤ Rx(θ[12]) ├┤ Rz(θ[16]) ├─░─┤ Rx(θ[18]) ├─░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░ └───────────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rx(θ[9]) ├─░─┤ Rx(θ[13]) ├┤ Rz(θ[17]) ├─░───────────────░─
         └──────────┘└──────────┘ ░ └──────────┘ ░ └───────────┘└───────────┘ ░               ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #8'):
        pv_list = list(ParameterVector(parameter_prefix, (5 * num_qubits - 1) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(2 * reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits - 1):
                if (r + i) % 2 == 0:
                    self.crx(pv_list.pop(0), i, i + 1)
            if insert_barriers and r != 2 * reps - 1:
                self.barrier()


class Circuit9(TwoLocal):
    """
         ┌───┐ ░                       ┌──────────┐
    q_0: ┤ H ├─░──■────────────────────┤ Rx(θ[0]) ├
         ├───┤ ░  │                    ├──────────┤
    q_1: ┤ H ├─░──■──────■─────────────┤ Rx(θ[1]) ├
         ├───┤ ░         │             ├──────────┤
    q_2: ┤ H ├─░─────────■──────■──────┤ Rx(θ[2]) ├
         ├───┤ ░                │      ├──────────┤
    q_3: ┤ H ├─░────────────────■──────┤ Rx(θ[3]) ├
         └───┘ ░                       └──────────┘
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False, skip_final_rotation_layer=True,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #9'):
        super().__init__(num_qubits, rotation_blocks='h', entanglement_blocks=['cz', 'rx'], entanglement='linear',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit10(TwoLocal):
    """
         ┌──────────┐ ░               ┌──────────┐
    q_0: ┤ Ry(θ[0]) ├─░──■──■─────────┤ Ry(θ[4]) ├
         ├──────────┤ ░  │  │         ├──────────┤
    q_1: ┤ Ry(θ[1]) ├─░──┼──■──■──────┤ Ry(θ[5]) ├
         ├──────────┤ ░  │     │      ├──────────┤
    q_2: ┤ Ry(θ[2]) ├─░──┼─────■──■───┤ Ry(θ[6]) ├
         ├──────────┤ ░  │        │   ├──────────┤
    q_3: ┤ Ry(θ[3]) ├─░──■────────■───┤ Ry(θ[7]) ├
         └──────────┘ ░               └──────────┘
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #10'):
        super().__init__(num_qubits, rotation_blocks='ry', entanglement_blocks='cz', entanglement='circular',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=False, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit11(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░       ░                                   ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░───■───░───────────────────────────────────░─
         ├──────────┤├──────────┤ ░ ┌─┴─┐ ░ ┌──────────┐┌───────────┐ ░       ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ X ├─░─┤ Rx(θ[8]) ├┤ Rz(θ[10]) ├─░───■───░─
         ├──────────┤├──────────┤ ░ └───┘ ░ ├──────────┤├───────────┤ ░ ┌─┴─┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░───■───░─┤ Rx(θ[9]) ├┤ Rz(θ[11]) ├─░─┤ X ├─░─
         ├──────────┤├──────────┤ ░ ┌─┴─┐ ░ └──────────┘└───────────┘ ░ └───┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ X ├─░───────────────────────────────────░─
         └──────────┘└──────────┘ ░ └───┘ ░                                   ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #11'):
        pv_list = list(ParameterVector(parameter_prefix, (4 * num_qubits - 4) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(2 * reps):
            if r % 2 == 0:
                for q in range(num_qubits):
                    self.rx(pv_list.pop(0), q)
                for q in range(num_qubits):
                    self.rz(pv_list.pop(0), q)
            else:
                for q in range(1, num_qubits - 1):
                    self.rx(pv_list.pop(0), q)
                for q in range(1, num_qubits - 1):
                    self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits - 1):
                if (r + i) % 2 == 0:
                    self.cx(i, i + 1)
            if insert_barriers and r != 2 * reps - 1:
                self.barrier()


class Circuit12(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░     ░                                 ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──■──░─────────────────────────────────░─
         ├──────────┤├──────────┤ ░  │  ░ ┌──────────┐┌───────────┐ ░     ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░──■──░─┤ Rx(θ[8]) ├┤ Rz(θ[10]) ├─░──■──░─
         ├──────────┤├──────────┤ ░     ░ ├──────────┤├───────────┤ ░  │  ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──■──░─┤ Rx(θ[9]) ├┤ Rz(θ[11]) ├─░──■──░─
         ├──────────┤├──────────┤ ░  │  ░ └──────────┘└───────────┘ ░     ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░──■──░─────────────────────────────────░─
         └──────────┘└──────────┘ ░     ░                                 ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #12'):
        pv_list = list(ParameterVector(parameter_prefix, (4 * num_qubits - 4) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(2 * reps):
            if r % 2 == 0:
                for q in range(num_qubits):
                    self.rx(pv_list.pop(0), q)
                for q in range(num_qubits):
                    self.rz(pv_list.pop(0), q)
            else:
                for q in range(1, num_qubits - 1):
                    self.rx(pv_list.pop(0), q)
                for q in range(1, num_qubits - 1):
                    self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits - 1):
                if (r + i) % 2 == 0:
                    self.cz(i, i + 1)
            if insert_barriers and r != 2 * reps - 1:
                self.barrier()


class Circuit13(TwoLocal):
    """
         ┌──────────┐ ░ ┌──────────┐                                     ░  ┌──────────┐ ░                           »
    q_0: ┤ Ry(θ[0]) ├─░─┤ Rz(θ[4]) ├─────■───────────────────────────────░──┤ Ry(θ[8]) ├─░────────────────────■──────»
         ├──────────┤ ░ └────┬─────┘┌────┴─────┐                         ░  ├──────────┤ ░                    │      »
    q_1: ┤ Ry(θ[1]) ├─░──────┼──────┤ Rz(θ[5]) ├─────■───────────────────░──┤ Ry(θ[9]) ├─░────────────────────┼──────»
         ├──────────┤ ░      │      └──────────┘┌────┴─────┐             ░ ┌┴──────────┤ ░ ┌───────────┐      │      »
    q_2: ┤ Ry(θ[2]) ├─░──────┼──────────────────┤ Rz(θ[6]) ├─────■───────░─┤ Ry(θ[10]) ├─░─┤ Rz(θ[12]) ├──────┼──────»
         ├──────────┤ ░      │                  └──────────┘┌────┴─────┐ ░ ├───────────┤ ░ └─────┬─────┘┌─────┴─────┐»
    q_3: ┤ Ry(θ[3]) ├─░──────■──────────────────────────────┤ Rz(θ[7]) ├─░─┤ Ry(θ[11]) ├─░───────■──────┤ Rz(θ[13]) ├»
         └──────────┘ ░                                     └──────────┘ ░ └───────────┘ ░              └───────────┘»
    «     ┌───────────┐              ░
    «q_0: ┤ Rz(θ[14]) ├──────────────░─
    «     └─────┬─────┘┌───────────┐ ░
    «q_1: ──────■──────┤ Rz(θ[15]) ├─░─
    «                  └─────┬─────┘ ░
    «q_2: ───────────────────■───────░─
    «                                ░
    «q_3: ───────────────────────────░─
    «                                ░
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #13'):
        super().__init__(num_qubits, rotation_blocks='ry', entanglement_blocks='crz', entanglement='sca',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=True, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit14(TwoLocal):
    """
         ┌──────────┐ ░ ┌──────────┐                                     ░  ┌──────────┐ ░                           »
    q_0: ┤ Ry(θ[0]) ├─░─┤ Rx(θ[4]) ├─────■───────────────────────────────░──┤ Ry(θ[8]) ├─░────────────────────■──────»
         ├──────────┤ ░ └────┬─────┘┌────┴─────┐                         ░  ├──────────┤ ░                    │      »
    q_1: ┤ Ry(θ[1]) ├─░──────┼──────┤ Rx(θ[5]) ├─────■───────────────────░──┤ Ry(θ[9]) ├─░────────────────────┼──────»
         ├──────────┤ ░      │      └──────────┘┌────┴─────┐             ░ ┌┴──────────┤ ░ ┌───────────┐      │      »
    q_2: ┤ Ry(θ[2]) ├─░──────┼──────────────────┤ Rx(θ[6]) ├─────■───────░─┤ Ry(θ[10]) ├─░─┤ Rx(θ[12]) ├──────┼──────»
         ├──────────┤ ░      │                  └──────────┘┌────┴─────┐ ░ ├───────────┤ ░ └─────┬─────┘┌─────┴─────┐»
    q_3: ┤ Ry(θ[3]) ├─░──────■──────────────────────────────┤ Rx(θ[7]) ├─░─┤ Ry(θ[11]) ├─░───────■──────┤ Rx(θ[13]) ├»
         └──────────┘ ░                                     └──────────┘ ░ └───────────┘ ░              └───────────┘»
    «     ┌───────────┐              ░
    «q_0: ┤ Rx(θ[14]) ├──────────────░─
    «     └─────┬─────┘┌───────────┐ ░
    «q_1: ──────■──────┤ Rx(θ[15]) ├─░─
    «                  └─────┬─────┘ ░
    «q_2: ───────────────────■───────░─
    «                                ░
    «q_3: ───────────────────────────░─
    «                                ░
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #14'):
        super().__init__(num_qubits, rotation_blocks='ry', entanglement_blocks='crx', entanglement='sca',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=True, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit15(TwoLocal):
    """
         ┌──────────┐ ░ ┌───┐                ░ ┌──────────┐ ░           ┌───┐      ░
    q_0: ┤ Ry(θ[0]) ├─░─┤ X ├──■─────────────░─┤ Ry(θ[4]) ├─░────────■──┤ X ├──────░─
         ├──────────┤ ░ └─┬─┘┌─┴─┐           ░ ├──────────┤ ░        │  └─┬─┘┌───┐ ░
    q_1: ┤ Ry(θ[1]) ├─░───┼──┤ X ├──■────────░─┤ Ry(θ[5]) ├─░────────┼────■──┤ X ├─░─
         ├──────────┤ ░   │  └───┘┌─┴─┐      ░ ├──────────┤ ░ ┌───┐  │       └─┬─┘ ░
    q_2: ┤ Ry(θ[2]) ├─░───┼───────┤ X ├──■───░─┤ Ry(θ[6]) ├─░─┤ X ├──┼─────────■───░─
         ├──────────┤ ░   │       └───┘┌─┴─┐ ░ ├──────────┤ ░ └─┬─┘┌─┴─┐           ░
    q_3: ┤ Ry(θ[3]) ├─░───■────────────┤ X ├─░─┤ Ry(θ[7]) ├─░───■──┤ X ├───────────░─
         └──────────┘ ░                └───┘ ░ └──────────┘ ░      └───┘           ░
    """

    def __init__(self, num_qubits=None, reps=3, skip_unentangled_qubits=False,
                 parameter_prefix='θ', insert_barriers=False, initial_state=None, name='Circuit #15'):
        super().__init__(num_qubits, rotation_blocks='ry', entanglement_blocks='cx', entanglement='sca',
                         reps=reps, skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=True, parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers, initial_state=initial_state, name=name)


class Circuit16(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░              ░               ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────░───────────────░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░               ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─░───────■───────░─
         ├──────────┤├──────────┤ ░ └──────────┘ ░ ┌─────┴─────┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■───────░─┤ Rx(θ[10]) ├─░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ └───────────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rz(θ[9]) ├─░───────────────░─
         └──────────┘└──────────┘ ░ └──────────┘ ░               ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #16'):
        pv_list = list(ParameterVector(parameter_prefix, (3 * num_qubits - 1) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits - 1):
                if i % 2 == 0:
                    self.crz(pv_list.pop(0), i, i + 1)
            for i in range(num_qubits - 1):
                if i % 2 == 1:
                    self.crz(pv_list.pop(0), i, i + 1)
            if insert_barriers and r != reps - 1:
                self.barrier()


class Circuit17(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░              ░               ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────░───────────────░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░               ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─░───────■───────░─
         ├──────────┤├──────────┤ ░ └──────────┘ ░ ┌─────┴─────┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■───────░─┤ Rx(θ[10]) ├─░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ └───────────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rx(θ[9]) ├─░───────────────░─
         └──────────┘└──────────┘ ░ └──────────┘ ░               ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #17'):
        pv_list = list(ParameterVector(parameter_prefix, (3 * num_qubits - 1) * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits - 1):
                if i % 2 == 0:
                    self.crx(pv_list.pop(0), i, i + 1)
            for i in range(num_qubits - 1):
                if i % 2 == 1:
                    self.crx(pv_list.pop(0), i, i + 1)
            if insert_barriers and r != reps - 1:
                self.barrier()


class Circuit18(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░             ┌──────────┐                           ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■──────┤ Rz(θ[9]) ├───────────────────────────░─
         ├──────────┤├──────────┤ ░      │      └────┬─────┘┌───────────┐              ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░──────┼───────────■──────┤ Rz(θ[10]) ├──────────────░─
         ├──────────┤├──────────┤ ░      │                  └─────┬─────┘┌───────────┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────┼────────────────────────■──────┤ Rz(θ[11]) ├─░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐                         └─────┬─────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rz(θ[8]) ├───────────────────────────────■───────░─
         └──────────┘└──────────┘ ░ └──────────┘                                       ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #18'):
        pv_list = list(ParameterVector(parameter_prefix, 3 * num_qubits * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits):
                self.crz(pv_list.pop(0), i % num_qubits, (i - 1) % num_qubits)
            if insert_barriers and r != reps - 1:
                self.barrier()


class Circuit19(QuantumCircuit):
    """
         ┌──────────┐┌──────────┐ ░             ┌──────────┐                           ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■──────┤ Rx(θ[9]) ├───────────────────────────░─
         ├──────────┤├──────────┤ ░      │      └────┬─────┘┌───────────┐              ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░──────┼───────────■──────┤ Rx(θ[10]) ├──────────────░─
         ├──────────┤├──────────┤ ░      │                  └─────┬─────┘┌───────────┐ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────┼────────────────────────■──────┤ Rx(θ[11]) ├─░─
         ├──────────┤├──────────┤ ░ ┌────┴─────┐                         └─────┬─────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rx(θ[8]) ├───────────────────────────────■───────░─
         └──────────┘└──────────┘ ░ └──────────┘                                       ░
    """

    def __init__(self, num_qubits=None, reps=3, parameter_prefix='θ', insert_barriers=False, initial_state=None,
                 name='Circuit #19'):
        pv_list = list(ParameterVector(parameter_prefix, 3 * num_qubits * reps))
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        if initial_state is not None:
            self.compose(initial_state, inplace=True)

        for r in range(reps):
            for q in range(num_qubits):
                self.rx(pv_list.pop(0), q)
            for q in range(num_qubits):
                self.rz(pv_list.pop(0), q)
            if insert_barriers:
                self.barrier()
            for i in range(num_qubits):
                self.crx(pv_list.pop(0), i % num_qubits, (i - 1) % num_qubits)
            if insert_barriers and r != reps - 1:
                self.barrier()


def sample_circuit(circuit_id: int):
    """
    ref: https://arxiv.org/abs/1905.10876v1


    Circuit #1
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░─
         ├──────────┤├──────────┤ ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─
         ├──────────┤├──────────┤ ░
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─
         ├──────────┤├──────────┤ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─
         └──────────┘└──────────┘ ░
    Circuit #2
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░───■────────────
         ├──────────┤├──────────┤ ░ ┌─┴─┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ X ├──■───────
         ├──────────┤├──────────┤ ░ └───┘┌─┴─┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────┤ X ├──■──
         ├──────────┤├──────────┤ ░      └───┘┌─┴─┐
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░───────────┤ X ├
         └──────────┘└──────────┘ ░           └───┘
    Circuit #3
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────────────────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─────■───────────────────
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rz(θ[9]) ├──────■──────
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rz(θ[10]) ├
         └──────────┘└──────────┘ ░                         └───────────┘
    Circuit #4
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────────────────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─────■───────────────────
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rx(θ[9]) ├──────■──────
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rx(θ[10]) ├
         └──────────┘└──────────┘ ░                         └───────────┘
    Circuit #5
         ┌──────────┐┌──────────┐ ░                                      ┌───────────┐                          »
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────■────────────■──────┤ Rz(θ[11]) ├──────────────────────────»
         ├──────────┤├──────────┤ ░ ┌────┴─────┐     │            │      └─────┬─────┘                          »
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─────┼────────────┼────────────■────────────■────────────■──────»
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐      │                   ┌─────┴─────┐      │      »
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rz(θ[9]) ├──────┼───────────────────┤ Rz(θ[12]) ├──────┼──────»
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐             └───────────┘┌─────┴─────┐»
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rz(θ[10]) ├──────────────────────────┤ Rz(θ[13]) ├»
         └──────────┘└──────────┘ ░                         └───────────┘                          └───────────┘»
    «     ┌───────────┐                          ┌───────────┐                           ░ ┌───────────┐┌───────────┐
    «q_0: ┤ Rz(θ[14]) ├──────────────────────────┤ Rz(θ[17]) ├───────────────────────────░─┤ Rx(θ[20]) ├┤ Rz(θ[24]) ├
    «     └─────┬─────┘┌───────────┐             └─────┬─────┘┌───────────┐              ░ ├───────────┤├───────────┤
    «q_1: ──────┼──────┤ Rz(θ[15]) ├───────────────────┼──────┤ Rz(θ[18]) ├──────────────░─┤ Rx(θ[21]) ├┤ Rz(θ[25]) ├
    «           │      └─────┬─────┘                   │      └─────┬─────┘┌───────────┐ ░ ├───────────┤├───────────┤
    «q_2: ──────■────────────■────────────■────────────┼────────────┼──────┤ Rz(θ[19]) ├─░─┤ Rx(θ[22]) ├┤ Rz(θ[26]) ├
    «                               ┌─────┴─────┐      │            │      └─────┬─────┘ ░ ├───────────┤├───────────┤
    «q_3: ──────────────────────────┤ Rz(θ[16]) ├──────■────────────■────────────■───────░─┤ Rx(θ[23]) ├┤ Rz(θ[27]) ├
    «                               └───────────┘                                        ░ └───────────┘└───────────┘
    Circuit #6
         ┌──────────┐┌──────────┐ ░                                      ┌───────────┐                          »
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────■────────────■──────┤ Rx(θ[11]) ├──────────────────────────»
         ├──────────┤├──────────┤ ░ ┌────┴─────┐     │            │      └─────┬─────┘                          »
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─────┼────────────┼────────────■────────────■────────────■──────»
         ├──────────┤├──────────┤ ░ └──────────┘┌────┴─────┐      │                   ┌─────┴─────┐      │      »
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░─────────────┤ Rx(θ[9]) ├──────┼───────────────────┤ Rx(θ[12]) ├──────┼──────»
         ├──────────┤├──────────┤ ░             └──────────┘┌─────┴─────┐             └───────────┘┌─────┴─────┐»
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─────────────────────────┤ Rx(θ[10]) ├──────────────────────────┤ Rx(θ[13]) ├»
         └──────────┘└──────────┘ ░                         └───────────┘                          └───────────┘»
    «     ┌───────────┐                          ┌───────────┐                           ░ ┌───────────┐┌───────────┐
    «q_0: ┤ Rx(θ[14]) ├──────────────────────────┤ Rx(θ[17]) ├───────────────────────────░─┤ Rx(θ[20]) ├┤ Rz(θ[24]) ├
    «     └─────┬─────┘┌───────────┐             └─────┬─────┘┌───────────┐              ░ ├───────────┤├───────────┤
    «q_1: ──────┼──────┤ Rx(θ[15]) ├───────────────────┼──────┤ Rx(θ[18]) ├──────────────░─┤ Rx(θ[21]) ├┤ Rz(θ[25]) ├
    «           │      └─────┬─────┘                   │      └─────┬─────┘┌───────────┐ ░ ├───────────┤├───────────┤
    «q_2: ──────■────────────■────────────■────────────┼────────────┼──────┤ Rx(θ[19]) ├─░─┤ Rx(θ[22]) ├┤ Rz(θ[26]) ├
    «                               ┌─────┴─────┐      │            │      └─────┬─────┘ ░ ├───────────┤├───────────┤
    «q_3: ──────────────────────────┤ Rx(θ[16]) ├──────■────────────■────────────■───────░─┤ Rx(θ[23]) ├┤ Rz(θ[27]) ├
    «                               └───────────┘                                        ░ └───────────┘└───────────┘
    Circuit #7
         ┌──────────┐┌──────────┐ ░              ░ ┌───────────┐┌───────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────░─┤ Rx(θ[10]) ├┤ Rz(θ[14]) ├─░──────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├─░─┤ Rx(θ[11]) ├┤ Rz(θ[15]) ├─░───────■──────
         ├──────────┤├──────────┤ ░ └──────────┘ ░ ├───────────┤├───────────┤ ░ ┌─────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■───────░─┤ Rx(θ[12]) ├┤ Rz(θ[16]) ├─░─┤ Rz(θ[18]) ├
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░ └───────────┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rz(θ[9]) ├─░─┤ Rx(θ[13]) ├┤ Rz(θ[17]) ├─░──────────────
         └──────────┘└──────────┘ ░ └──────────┘ ░ └───────────┘└───────────┘ ░
    Circuit #8
         ┌──────────┐┌──────────┐ ░              ░ ┌───────────┐┌───────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────░─┤ Rx(θ[10]) ├┤ Rz(θ[14]) ├─░──────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├─░─┤ Rx(θ[11]) ├┤ Rz(θ[15]) ├─░───────■──────
         ├──────────┤├──────────┤ ░ └──────────┘ ░ ├───────────┤├───────────┤ ░ ┌─────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■───────░─┤ Rx(θ[12]) ├┤ Rz(θ[16]) ├─░─┤ Rx(θ[18]) ├
         ├──────────┤├──────────┤ ░ ┌────┴─────┐ ░ ├───────────┤├───────────┤ ░ └───────────┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rx(θ[9]) ├─░─┤ Rx(θ[13]) ├┤ Rz(θ[17]) ├─░──────────────
         └──────────┘└──────────┘ ░ └──────────┘ ░ └───────────┘└───────────┘ ░
    Circuit #9
         ┌───┐ ░    ┌──────────┐
    q_0: ┤ H ├─░──■─┤ Rx(θ[0]) ├────────────────────────
         ├───┤ ░  │ └──────────┘┌──────────┐
    q_1: ┤ H ├─░──■──────■──────┤ Rx(θ[1]) ├────────────
         ├───┤ ░         │      └──────────┘┌──────────┐
    q_2: ┤ H ├─░─────────■───────────■──────┤ Rx(θ[2]) ├
         ├───┤ ░                     │      ├──────────┤
    q_3: ┤ H ├─░─────────────────────■──────┤ Rx(θ[3]) ├
         └───┘ ░                            └──────────┘
    Circuit #10
         ┌──────────┐ ░              ░ ┌──────────┐
    q_0: ┤ Ry(θ[0]) ├─░──■──■────────░─┤ Ry(θ[4]) ├
         ├──────────┤ ░  │  │        ░ ├──────────┤
    q_1: ┤ Ry(θ[1]) ├─░──┼──■──■─────░─┤ Ry(θ[5]) ├
         ├──────────┤ ░  │     │     ░ ├──────────┤
    q_2: ┤ Ry(θ[2]) ├─░──┼─────■──■──░─┤ Ry(θ[6]) ├
         ├──────────┤ ░  │        │  ░ ├──────────┤
    q_3: ┤ Ry(θ[3]) ├─░──■────────■──░─┤ Ry(θ[7]) ├
         └──────────┘ ░              ░ └──────────┘
    Circuit #11
         ┌──────────┐┌──────────┐ ░       ░                           ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░───■───░───────────────────────────░──────
         ├──────────┤├──────────┤ ░ ┌─┴─┐ ░ ┌──────────┐┌───────────┐ ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ X ├─░─┤ Rx(θ[8]) ├┤ Rz(θ[10]) ├─░───■──
         ├──────────┤├──────────┤ ░ └───┘ ░ ├──────────┤├───────────┤ ░ ┌─┴─┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░───■───░─┤ Rx(θ[9]) ├┤ Rz(θ[11]) ├─░─┤ X ├
         ├──────────┤├──────────┤ ░ ┌─┴─┐ ░ └──────────┘└───────────┘ ░ └───┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ X ├─░───────────────────────────░──────
         └──────────┘└──────────┘ ░ └───┘ ░                           ░
    Circuit #12
         ┌──────────┐┌──────────┐ ░     ░                           ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──■──░───────────────────────────░────
         ├──────────┤├──────────┤ ░  │  ░ ┌──────────┐┌───────────┐ ░
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░──■──░─┤ Rx(θ[8]) ├┤ Rz(θ[10]) ├─░──■─
         ├──────────┤├──────────┤ ░     ░ ├──────────┤├───────────┤ ░  │
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──■──░─┤ Rx(θ[9]) ├┤ Rz(θ[11]) ├─░──■─
         ├──────────┤├──────────┤ ░  │  ░ └──────────┘└───────────┘ ░
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░──■──░───────────────────────────░────
         └──────────┘└──────────┘ ░     ░                           ░
    Circuit #13
         ┌──────────┐ ░ ┌──────────┐
    q_0: ┤ Ry(θ[0]) ├─░─┤ Rz(θ[4]) ├─────■──────────────────────────────
         ├──────────┤ ░ └────┬─────┘┌────┴─────┐
    q_1: ┤ Ry(θ[1]) ├─░──────┼──────┤ Rz(θ[5]) ├─────■──────────────────
         ├──────────┤ ░      │      └──────────┘┌────┴─────┐
    q_2: ┤ Ry(θ[2]) ├─░──────┼──────────────────┤ Rz(θ[6]) ├─────■──────
         ├──────────┤ ░      │                  └──────────┘┌────┴─────┐
    q_3: ┤ Ry(θ[3]) ├─░──────■──────────────────────────────┤ Rz(θ[7]) ├
         └──────────┘ ░                                     └──────────┘
    Circuit #14
         ┌──────────┐ ░ ┌──────────┐
    q_0: ┤ Ry(θ[0]) ├─░─┤ Rx(θ[4]) ├─────■──────────────────────────────
         ├──────────┤ ░ └────┬─────┘┌────┴─────┐
    q_1: ┤ Ry(θ[1]) ├─░──────┼──────┤ Rx(θ[5]) ├─────■──────────────────
         ├──────────┤ ░      │      └──────────┘┌────┴─────┐
    q_2: ┤ Ry(θ[2]) ├─░──────┼──────────────────┤ Rx(θ[6]) ├─────■──────
         ├──────────┤ ░      │                  └──────────┘┌────┴─────┐
    q_3: ┤ Ry(θ[3]) ├─░──────■──────────────────────────────┤ Rx(θ[7]) ├
         └──────────┘ ░                                     └──────────┘
    Circuit #15
         ┌──────────┐ ░ ┌───┐
    q_0: ┤ Ry(θ[0]) ├─░─┤ X ├──■────────────
         ├──────────┤ ░ └─┬─┘┌─┴─┐
    q_1: ┤ Ry(θ[1]) ├─░───┼──┤ X ├──■───────
         ├──────────┤ ░   │  └───┘┌─┴─┐
    q_2: ┤ Ry(θ[2]) ├─░───┼───────┤ X ├──■──
         ├──────────┤ ░   │       └───┘┌─┴─┐
    q_3: ┤ Ry(θ[3]) ├─░───■────────────┤ X ├
         └──────────┘ ░                └───┘
    Circuit #16
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rz(θ[8]) ├──────■──────
         ├──────────┤├──────────┤ ░ └──────────┘┌─────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■──────┤ Rz(θ[10]) ├
         ├──────────┤├──────────┤ ░ ┌────┴─────┐└───────────┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rz(θ[9]) ├─────────────
         └──────────┘└──────────┘ ░ └──────────┘
    Circuit #17
         ┌──────────┐┌──────────┐ ░
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■───────────────────
         ├──────────┤├──────────┤ ░ ┌────┴─────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ Rx(θ[8]) ├──────■──────
         ├──────────┤├──────────┤ ░ └──────────┘┌─────┴─────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────■──────┤ Rx(θ[10]) ├
         ├──────────┤├──────────┤ ░ ┌────┴─────┐└───────────┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rx(θ[9]) ├─────────────
         └──────────┘└──────────┘ ░ └──────────┘
    Circuit #18
         ┌──────────┐┌──────────┐ ░             ┌──────────┐
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■──────┤ Rz(θ[9]) ├──────────────────────────
         ├──────────┤├──────────┤ ░      │      └────┬─────┘┌───────────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░──────┼───────────■──────┤ Rz(θ[10]) ├─────────────
         ├──────────┤├──────────┤ ░      │                  └─────┬─────┘┌───────────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────┼────────────────────────■──────┤ Rz(θ[11]) ├
         ├──────────┤├──────────┤ ░ ┌────┴─────┐                         └─────┬─────┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rz(θ[8]) ├───────────────────────────────■──────
         └──────────┘└──────────┘ ░ └──────────┘
    Circuit #19
         ┌──────────┐┌──────────┐ ░             ┌──────────┐
    q_0: ┤ Rx(θ[0]) ├┤ Rz(θ[4]) ├─░──────■──────┤ Rx(θ[9]) ├──────────────────────────
         ├──────────┤├──────────┤ ░      │      └────┬─────┘┌───────────┐
    q_1: ┤ Rx(θ[1]) ├┤ Rz(θ[5]) ├─░──────┼───────────■──────┤ Rx(θ[10]) ├─────────────
         ├──────────┤├──────────┤ ░      │                  └─────┬─────┘┌───────────┐
    q_2: ┤ Rx(θ[2]) ├┤ Rz(θ[6]) ├─░──────┼────────────────────────■──────┤ Rx(θ[11]) ├
         ├──────────┤├──────────┤ ░ ┌────┴─────┐                         └─────┬─────┘
    q_3: ┤ Rx(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ Rx(θ[8]) ├───────────────────────────────■──────
         └──────────┘└──────────┘ ░ └──────────┘

    """
    return getattr(sys.modules[__name__], f"Circuit{circuit_id}")


class SingleQubitFeatureMap(QuantumCircuit):
    def __init__(self, num_qubits=None, reps=1, parameter_prefix='X', name='SingleQubitFeatureMap'):
        super().__init__(QuantumRegister(size=num_qubits, name='q'), name=name)
        parameters = ParameterVector(parameter_prefix, 2)
        for _ in range(reps):
            for n in range(num_qubits):
                self.ry(parameters[0], n)
                self.rz(parameters[1], n)


class MultilayerCircuit9FeatureMap(Circuit9):
    def __init__(self, num_qubits=None, reps=2, parameter_prefix='X', name='MultilayerCircuit9FeatureMap'):
        super().__init__(num_qubits, reps=1, parameter_prefix=parameter_prefix)
        featur_map1 = Circuit9(num_qubits, reps=1, parameter_prefix=parameter_prefix)
        for _ in range(reps):
            self.compose(featur_map1, inplace=True)


if __name__ == '__main__':
    for i in range(1, 19 + 1):
        qc = globals()[f"Circuit{i}"](4, reps=1, insert_barriers=True)
        print(qc.name)
        if isinstance(qc, TwoLocal):
            qc = qc.decompose()
        print(qc.draw(fold=120))
        print('\n')
