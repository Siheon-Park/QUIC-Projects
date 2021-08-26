import logging
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Parameter
from qiskit.circuit.library import RXGate, RYGate, RZGate, XGate
from qiskit.utils import QuantumInstance
from qiskit import transpile
from . import QuantumClassifier, postprocess_Z_expectation
from typing import Union, List, Iterable, Dict

from qiskit_machine_learning.kernels import QuantumKernel

from qiskit.quantum_info.operators import Operator, Pauli

from .qasvm import ParameterArray
from . import qasvm

logger = logging.getLogger(__name__)
logger.warning(f"DEPRECATED: {__name__} will be removed")


class NormQSVM(QuantumClassifier):
    def __init__(self, data: np.ndarray, label: np.ndarray,
                 quantum_instance: QuantumInstance, lamda: float = 1.0,
                 feature_map: QuantumCircuit = None, var_form: QuantumCircuit = None,
                 initial_point: np.ndarray = None):
        logger.warning(f"DEPRECATED: use {qasvm.NormQSVM} instead")
        super().__init__(data, label)
        del self.alpha
        self.polary = 2 * self.label - 1
        self.quantum_instance = quantum_instance
        self.feature_map = feature_map
        self.var_form = var_form
        self.lamda = lamda

        self._parameters = ParameterArray(self.var_form.parameters)
        self.num_parameters = len(self.parameters)

        if initial_point is None:
            self.initial_point = np.pi * (2 * np.random.random(self.num_parameters) - 1)
        else:
            self.initial_point = initial_point
        self.parameters.update(self.initial_point)
        self.test_data_params = ParameterVector('x', self.dim_data)
        self.first_order_circuit, self.second_order_cirucit = self._create_circuits()
        self.transpiled_first_order_circuit = self.quantum_instance.transpile(self.first_order_circuit)[0]
        self.transpiled_second_order_circuit = self.quantum_instance.transpile(self.second_order_cirucit)[0]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, new_params: np.ndarray):
        self._parameters.update(new_params)

    def f(self, testdata: np.ndarray):
        def g(num: int):
            ret = self._evaluate_first_order_circuit(self.parameters, testdata)
            return ret['ayk'] + ret['ay'] / self.lamda

        return sum(map(g, range(10))) / 10

    def cost_fn(self, params: np.ndarray):
        def g(num: int):
            ret = self._evaluate_second_order_circuit(params)
            # return ret['aayyk'] + ret['aayy'] / self.lamda
            return ret['aayyk']

        return sum(map(g, range(10))) / 10

    def _create_circuits(self):
        num_index_qubits = self.var_form.num_qubits
        num_data_qubits = self.feature_map.num_qubits
        reg_dict = dict(map(lambda reg: (reg.name, reg), [
            QuantumRegister(1, 'a'), QuantumRegister(num_index_qubits, 'i'), QuantumRegister(num_data_qubits, 'xi'),
            QuantumRegister(1, 'yi'), ClassicalRegister(2, 'c2'), QuantumRegister(num_index_qubits, 'j'),
            QuantumRegister(num_data_qubits, 'xj'), QuantumRegister(1, 'yj'), ClassicalRegister(3, 'c3')
        ]))
        first_order_circuit = QuantumCircuit(*list(map(lambda key: reg_dict[key], ['a', 'i', 'xi', 'yi', 'xj', 'c2'])),
                                             name='1st_circuit')
        second_order_circuit = QuantumCircuit(
            *list(map(lambda key: reg_dict[key], ['a', 'i', 'xi', 'yi', 'j', 'xj', 'yj', 'c3'])), name='2nd_circuit')

        UD = _UDCirc(num_index_qubits, self.feature_map, self.data, self.label)
        UX = _UXCirc(self.feature_map, datum=self.test_data_params)

        first_order_circuit.compose(self.var_form, qubits=reg_dict['i'], inplace=True, wrap=True)
        first_order_circuit.compose(UD, qubits=[*reg_dict['i'], *reg_dict['xi'], *reg_dict['yi']], inplace=True,
                                    wrap=True)
        first_order_circuit.compose(UX, qubits=reg_dict['xj'], inplace=True, wrap=True)
        first_order_circuit.compose(_CSWAPCirc(num_data_qubits), [*reg_dict['a'], *reg_dict['xi'], *reg_dict['xj']],
                                    inplace=True, wrap=False)
        first_order_circuit.measure([*reg_dict['a'], *reg_dict['yi']], reg_dict['c2'])

        second_order_circuit.compose(self.var_form, qubits=reg_dict['i'], inplace=True, wrap=True)
        second_order_circuit.compose(self.var_form, qubits=reg_dict['j'], inplace=True, wrap=True)
        second_order_circuit.compose(UD, qubits=[*reg_dict['i'], *reg_dict['xi'], *reg_dict['yi']], inplace=True,
                                     wrap=True)
        second_order_circuit.compose(UD, qubits=[*reg_dict['j'], *reg_dict['xj'], *reg_dict['yj']], inplace=True,
                                     wrap=True)
        second_order_circuit.compose(_CSWAPCirc(num_data_qubits), [*reg_dict['a'], *reg_dict['xi'], *reg_dict['xj']],
                                     inplace=True, wrap=False)
        second_order_circuit.measure([*reg_dict['a'], *reg_dict['yi'], *reg_dict['yj']], reg_dict['c3'])
        return first_order_circuit, second_order_circuit

    def _evaluate_second_order_circuit(self, param: np.ndarray) -> Dict[str, float]:
        assert len(param) == self.num_parameters
        parameter = self.parameters.copy()
        parameter.update(param)
        _dict = self.quantum_instance.execute(
            [self.transpiled_second_order_circuit.assign_parameters(parameter.to_dict())], True).get_counts()
        eval_dict = dict()
        eval_dict['aayyk'] = postprocess_Z_expectation(3, _dict, 2, 1, 0)
        eval_dict['aayy'] = postprocess_Z_expectation(3, _dict, 1, 0)
        return eval_dict

    def _evaluate_first_order_circuit(self, param: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        assert len(param) == self.num_parameters
        parameter = self.parameters.copy()
        parameter.update(param)
        param_dict = parameter.to_dict()
        circ = self.transpiled_first_order_circuit.assign_parameters(param_dict)
        if len(data.shape) == 1:
            qc_list = [circ.assign_parameters(dict(zip(self.test_data_params, data)))]
        else:
            qc_list = [circ.assign_parameters(dict(zip(self.test_data_params, datum))) for datum in data]
        _dict = self.quantum_instance.execute(qc_list).get_counts()
        eval_dict = dict()
        if isinstance(_dict, dict):
            eval_dict['ayk'] = postprocess_Z_expectation(2, _dict, 1, 0)
            eval_dict['ay'] = postprocess_Z_expectation(2, _dict, 0)
        else:  # isinstance(_dict, list):
            eval_dict['ayk'] = np.array([postprocess_Z_expectation(2, __dict, 1, 0) for __dict in _dict])
            eval_dict['ay'] = np.array([postprocess_Z_expectation(2, __dict, 0) for __dict in _dict])
        return eval_dict


class _UDCirc(QuantumCircuit):
    def __init__(self, num_ctrl_qubits: int, feature_map: QuantumCircuit, data: np.ndarray, label: np.ndarray):
        assert len(data) == len(label)
        assert len(data) == 2 ** num_ctrl_qubits
        assert feature_map.num_parameters == data.shape[1]
        ctrl_reg = QuantumRegister(num_ctrl_qubits)
        target_xreg = QuantumRegister(feature_map.num_qubits)
        target_yreg = QuantumRegister(1)
        super().__init__(ctrl_reg, target_xreg, target_yreg, name=f'U({feature_map.name})',
                         global_phase=feature_map.global_phase, metadata=feature_map.metadata)
        feature_map = transpile(circuits=feature_map, basis_gates=['rx', 'ry', 'rz', 'cx'])
        feature_map = QuantumCircuit(target_xreg).compose(feature_map)
        data_list = [feature_map.assign_parameters(dict(zip(feature_map.parameters, datum))) for datum in data]
        for i, (gate, qubits, cbits) in enumerate(feature_map):
            if isinstance(gate, RXGate):
                self.ucrx([float(data_list[n][i][0].params[0]) for n in range(len(data))], ctrl_reg, qubits)
            elif isinstance(gate, RYGate):
                self.ucry([float(data_list[n][i][0].params[0]) for n in range(len(data))], ctrl_reg, qubits)
            elif isinstance(gate, RZGate):
                self.ucrz([float(data_list[n][i][0].params[0]) for n in range(len(data))], ctrl_reg, qubits)
            else:
                self.append(gate, qubits, cbits)
        self.ucrx(list(np.where(label > 0.5, 0, np.pi)), ctrl_reg, target_yreg)


class _UXCirc(QuantumCircuit):
    def __init__(self, feature_map: QuantumCircuit, datum: np.ndarray, transpiled: bool = False):
        assert len(datum) == feature_map.num_parameters
        target_xreg = QuantumRegister(feature_map.num_qubits)
        super().__init__(target_xreg, name=feature_map.name, global_phase=feature_map.global_phase,
                         metadata=feature_map.metadata)
        self.compose(feature_map.assign_parameters(dict(zip(feature_map.parameters, datum))), qubits=target_xreg,
                     inplace=True)


class _CSWAPCirc(QuantumCircuit):
    def __init__(self, num_swaps: int):
        ancila = QuantumRegister(1)
        target1 = QuantumRegister(num_swaps)
        target2 = QuantumRegister(num_swaps)
        super().__init__(ancila, target1, target2, name='C-SWAP(s)')
        for q1, q2 in zip(target1, target2):
            self.cswap(ancila, q1, q2)


class ProjectionOP(Operator):
    def __init__(self, i: int, dim: int):
        data = np.zeros((dim, dim))
        data[i][i] = 1
        super().__init__(data)


class FeatureMapOp(Operator):
    def __init__(self, feature_map: QuantumCircuit, datum: np.ndarray):
        super(FeatureMapOp, self).__init__(feature_map.assign_parameters(dict(zip(feature_map.parameters, datum))))


class ZeroOp(Operator):
    def __init__(self, dim: int):
        super().__init__(np.zeros((dim, dim)))


class _UDOPCirc(QuantumCircuit):
    def __init__(self, num_ctrl_qubits: int, feature_map: QuantumCircuit, data: np.ndarray, label: np.ndarray):
        assert len(data) == len(label)
        assert len(data) == 2 ** num_ctrl_qubits
        assert feature_map.num_parameters == data.shape[1]
        ctrl_reg = QuantumRegister(num_ctrl_qubits)
        target_xreg = QuantumRegister(feature_map.num_qubits)
        target_yreg = QuantumRegister(1)
        super().__init__(ctrl_reg, target_xreg, target_yreg, name=f'U({feature_map.name})',
                         global_phase=feature_map.global_phase, metadata=feature_map.metadata)
        UD = ZeroOp(2 ** num_ctrl_qubits) ^ ZeroOp(2 ** feature_map.num_qubits) ^ ZeroOp(2)
        for i, datum in enumerate(data):
            yy = Pauli('I') if label[i] > 0.5 else Pauli('X')
            UD += (ProjectionOP(i, 2 ** num_ctrl_qubits) ^ FeatureMapOp(feature_map, datum) ^ yy)

        self.append(UD, [*ctrl_reg, *target_xreg, *target_yreg])


class _UDNaiveCirc(QuantumCircuit):
    def __init__(self, num_ctrl_qubits: int, feature_map: QuantumCircuit, data: np.ndarray, label: np.ndarray):
        assert len(data) == len(label)
        assert len(data) == 2 ** num_ctrl_qubits
        assert feature_map.num_parameters == data.shape[1]
        ctrl_reg = QuantumRegister(num_ctrl_qubits)
        target_xreg = QuantumRegister(feature_map.num_qubits)
        target_yreg = QuantumRegister(1)
        super().__init__(ctrl_reg, target_xreg, target_yreg, name=f'U({feature_map.name})',
                         global_phase=feature_map.global_phase, metadata=feature_map.metadata)

        for ctrl_state in range(2 ** num_ctrl_qubits):
            feature_map_param_dict = dict(zip(feature_map.parameters, data[ctrl_state]))
            bind_feature_map = feature_map.assign_parameters(feature_map_param_dict)
            self.append(bind_feature_map.to_gate().control(ctrl_reg.size, ctrl_state=ctrl_state),
                        qargs=[*ctrl_reg, *target_xreg])
            if label[ctrl_state] < 0.5:
                self.append(XGate().control(ctrl_reg.size, ctrl_state=ctrl_state), qargs=[*ctrl_reg, *target_yreg])
