import logging
import numpy as np
from qiskit import transpile
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, U3Gate, RXGate, RYGate, RZGate
from typing import Union, List

logger = logging.getLogger(__name__)


# noinspection SpellCheckingInspection
class QASVM_circuit(QuantumCircuit):
    """
            Modulation of QASVM circuit (first order, second order) construction - Base Class:
            Example:
                qc = QASVM_circuit(num_data, dim_data, ord)
                qc.add_var_form(var_form)
                qc.UD_encode(feature_map, _feature_map_params, training_data=data, training_label=label, N=num_data)
                qc.X_encode(feature_map, _feature_map_params, testdata=testdata)
                qc.SWAP_test()
                qc.Z_expectation_measurement()
        """

    def __init__(self, num_index_qubits: int, num_data_qubits: int, order: int) -> None:
        assert (order == 1 or order == 2)
        super().__init__()
        regs = [QuantumRegister(1, 'a'), QuantumRegister(num_index_qubits, 'i'), QuantumRegister(num_data_qubits, 'xi'),
                QuantumRegister(1, 'yi')]
        if order == 1:
            regs.append(QuantumRegister(num_data_qubits, 'xj'))
            regs.append(ClassicalRegister(2, 'c'))
        else:
            regs.append(QuantumRegister(num_index_qubits, 'j'))
            regs.append(QuantumRegister(num_data_qubits, 'xj'))
            regs.append(QuantumRegister(1, 'yj'))
            regs.append(ClassicalRegister(3, 'c'))
        super().__init__(*regs)
        self._reg_dict = {r.name: r for r in regs}

    def SWAP_test(self):
        self.h(self._reg_dict['a'])
        [self.cswap(self._reg_dict['a'], self._reg_dict['xi'][d], self._reg_dict['xj'][d]) for d in
         range(len(self._reg_dict['xi']))]
        self.h(self._reg_dict['a'])

    def Z_expectation_measurement(self):
        self.measure(list(self._reg_dict['a']) + list(self._reg_dict['yi']) + list(self._reg_dict.get('yj', [])),
                     list(self._reg_dict['c']))

    def UD_encode(self, feature_map, feature_map_params, training_data: np.ndarray, training_label: np.ndarray, N: int,
                  reg: str = 'i'):
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x' + reg]
        target_y = self._reg_dict['y' + reg]
        for ctrl_state in range(N):
            feature_map_param_dict = {p: v for p, v in zip(feature_map_params, training_data[ctrl_state])}
            bind_feature_map = feature_map.assign_parameters(feature_map_param_dict)
            self.append(bind_feature_map.to_gate().control(len(ctrl_q), ctrl_state=ctrl_state),
                        qargs=list(ctrl_q) + list(target_q))
            if training_label[ctrl_state] < 0.5:
                self.append(XGate().control(len(ctrl_q), ctrl_state=ctrl_state), qargs=list(ctrl_q) + list(target_y))

    def X_encode(self, feature_map: QuantumCircuit, feature_map_params,
                 testdata: Union[np.ndarray, ParameterVector, List[Parameter]], reg: str = 'xj'):
        feature_map_param_dict = {p: v for p, v in zip(feature_map_params, testdata)}
        self.compose(feature_map.assign_parameters(feature_map_param_dict), qubits=list(self._reg_dict[reg]),
                     inplace=True)

    def add_var_form(self, var_form, reg: str = 'i'):
        self.compose(var_form, qubits=list(self._reg_dict[reg]), inplace=True)


class uniform_QASVM_circuit(QASVM_circuit):
    """ not using var_form """

    def add_var_form(self, var_form, reg: str = 'i'):
        if var_form is not None:
            logger.warning("ignoring {:} for {:} is not using it.".format('var_form', self.__class__.__name__))
        self.h(self._reg_dict[reg])


# noinspection SpellCheckingInspection
class Bloch_sphere_QASVM_circuit(QASVM_circuit):
    """ not using feature_map, feature_map_params """

    def UD_encode(self, feature_map, feature_map_params, training_data: np.ndarray, training_label: np.ndarray, N: int,
                  reg: str = 'i'):
        if feature_map is not None:
            logger.warning("ignoring {:} for {:} is not using it.".format('feature_map', self.__class__.__name__))
        if feature_map_params is not None:
            logger.debug("ignoring {:} for {:} is not using it.".format('feature_map_params', self.__class__.__name__))
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x' + reg]
        target_y = self._reg_dict['y' + reg]
        # self.uc([U3Gate(theta, phi, 0).to_matrix() for theta, phi in training_data], ctrl_q, target_q)
        self.ucry([theta for theta in training_data[:, 0]], ctrl_q, target_q)
        self.ucrz([phi for phi in training_data[:, 1]], ctrl_q, target_q)
        self.ucrx(list(np.where(training_label > 0.5, 0, np.pi)), ctrl_q, target_y)

    def X_encode(self, feature_map: QuantumCircuit, feature_map_params,
                 testdata: Union[np.ndarray, ParameterVector, List[Parameter]], reg: str = 'xj'):
        if feature_map is not None:
            logger.warning("ignoring {:} for {:} is not using it.".format('feature_map', self.__class__.__name__))
        if feature_map_params is not None:
            logger.debug("ignoring {:} for {:} is not using it.".format('feature_map_params', self.__class__.__name__))
        temp_feature_map = QuantumCircuit(1)
        temp_feature_map.u3(testdata[0], testdata[1], 0, 0)
        self.compose(temp_feature_map, qubits=list(self._reg_dict[reg]), inplace=True)


class Bloch_uniform_QASVM_circuit(Bloch_sphere_QASVM_circuit):
    """ not using var_form, feature_map, feature_map_params """
    add_var_form = uniform_QASVM_circuit.add_var_form


class _uc_QASVM_circuit(QASVM_circuit):
    def UD_encode(self, feature_map, feature_map_params, training_data: np.ndarray, training_label: np.ndarray, N: int,
                  reg: str = 'i'):
        if feature_map is not None:
            logger.warning("ignoring {:} for {:} is not using it.".format('feature_map', self.__class__.__name__))
        if feature_map_params is not None:
            logger.debug("ignoring {:} for {:} is not using it.".format('feature_map_params', self.__class__.__name__))
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x' + reg]
        target_y = self._reg_dict['y' + reg]
        self.uc([U3Gate(theta, phi, 0).to_matrix() for theta, phi in training_data], ctrl_q, target_q)
        self.ucrx(list(np.where(training_label > 0.5, 0, np.pi)), ctrl_q, target_y)


_uc_QASVM_circuit.X_encode = Bloch_sphere_QASVM_circuit.X_encode


# noinspection PyUnresolvedReferences
class NqSVM_circuit(QASVM_circuit):
    def UD_encode(self, feature_map, feature_map_params, training_data: np.ndarray, training_label: np.ndarray, N: int,
                  reg: str = 'i'):
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x' + reg]
        target_y = self._reg_dict['y' + reg]
        feature_map = transpile(feature_map, basis_gates=['rx', 'ry', 'rz', 'cx'])
        feature_map = transpile(circuits=feature_map, basis_gates=['rx', 'ry', 'rz', 'cx'])
        feature_map = QuantumCircuit(target_q).compose(feature_map)
        data_list = [feature_map.assign_parameters(dict(zip(feature_map.parameters, datum))) for datum in training_data]
        for i, (gate, qubits, cbits) in enumerate(feature_map):
            if isinstance(gate, RXGate):
                self.ucrx([float(data_list[n][i][0].params[0]) for n in range(len(training_data))], ctrl_q, qubits)
            elif isinstance(gate, RYGate):
                self.ucry([float(data_list[n][i][0].params[0]) for n in range(len(training_data))], ctrl_q, qubits)
            elif isinstance(gate, RZGate):
                self.ucrz([float(data_list[n][i][0].params[0]) for n in range(len(training_data))], ctrl_q, qubits)
            else:
                self.append(gate, qubits, cbits)
        self.ucrx(list(np.where(training_label > 0.5, 0, np.pi)), ctrl_q, target_y)


# global variables
_CIRCUIT_CLASS_DICT = {
    'QASVM': QASVM_circuit,
    'Bloch_sphere': Bloch_sphere_QASVM_circuit,
    'uniform': uniform_QASVM_circuit,
    'Bloch_uniform': Bloch_uniform_QASVM_circuit,
    '_uc': _uc_QASVM_circuit,
    'NqSVM': NqSVM_circuit
}


# noinspection SpellCheckingInspection
class AnsatzCircuit9(QuantumCircuit):
    """ ref : https://arxiv.org/pdf/1905.10876.pdf """

    def __init__(self, n, reps: int = 3, rotational_block: str = 'rx', entangling_block: str = 'cz',
                 parameter_prefix: str = 'θ'):
        pv = ParameterVector(parameter_prefix, length=n * reps)
        super().__init__(n, name='Circuit # 9')
        k = 0
        for i in range(reps):
            self.h(range(n))
            for j in range(n - 1):
                if entangling_block == 'cz':
                    self.cz(j, j + 1)
                elif entangling_block == 'cx':
                    self.cx(j, j + 1)
                elif entangling_block == 'cy':
                    self.cy(j, j + 1)
                else:
                    raise NameError()
            for l in range(n):
                if rotational_block == 'rx':
                    self.rx(pv[k], l)
                    k += 1
                elif rotational_block == 'ry':
                    self.ry(pv[k], l)
                    k += 1
                elif rotational_block == 'rz':
                    self.rz(pv[k], l)
                    k += 1
                else:
                    raise NameError()
