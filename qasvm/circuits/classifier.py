
import numpy as np
import math
from itertools import product
from numpy.lib.function_base import append

from qiskit.aqua.aqua_error import AquaError
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector
from ..classifier import BinarySVM, Classifier
from qasvm.kernel import Kernel
from qiskit import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit.library import XGate, TwoLocal
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm, ClassicalAlgorithm
from qiskit.aqua.algorithms.classifiers.vqc import VQAlgorithm
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.feature_maps import FeatureMap, RawFeatureVector
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.providers import Backend, BaseBackend
from qiskit.providers.aer import QasmSimulator, UnitarySimulator
from typing import Union, Optional, Dict, List

class BinaryQASVM(QuantumAlgorithm):
    def __init__(self, 
                optimizer: Optimizer,
                feature_map: Union[QuantumCircuit, FeatureMap],
                var_form: Union[QuantumCircuit, VariationalForm],
                training_dataset: Optional[Dict[str, np.ndarray]] = None,
                test_dataset: Optional[Dict[str, np.ndarray]] = None,
                quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        super().__init__(quantum_instance)
        self.optimizer = optimizer
        self.feature_map = feature_map
        self.var_form = var_form
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.quantum_instance = quantum_instance
        self._validate_datset(self.training_dataset)

        self.primal_qcs = [self._create_primal_circuit(testdata) for testdata in self.data]
        self.dual_qc = self._create_dual_circuit()

    def _create_primal_circuit(self, testdata:np.ndarray):
        qr_a = QuantumRegister(1, 'a')
        qr_i = QuantumRegister(int(math.log2(self.num_data)), 'i')
        qr_xi = QuantumRegister(int(math.log2(self.dim_data)), 'x')
        qr_yi = QuantumRegister(1, 'y')
        qr_xt = QuantumRegister(len(qr_xi), 'xt')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr_a, qr_i, qr_xi, qr_yi, qr_xt, cr)
        qc.compose(self.var_form, qubits=list(qr_i), inplace=True)
        qc.ucrx()
        for ctrl_state in range(self.num_data):
            bind_feature_map = self.feature_map.bind_parameters({tuple(self.feature_map.parameters)[d]:x for d, x in enumerate(self.data[ctrl_state])})
            qc.append(bind_feature_map.to_gate().control(len(qr_i), ctrl_state=ctrl_state), qargs=list(qr_i)+list(qr_xi))
            if self.label[ctrl_state]==1:
                qc.append(XGate().control(len(qr_i), ctrl_state=ctrl_state), qargs=list(qr_i)+list(qr_yi))
        qc.append(self.feature_map.bind_parameters({tuple(self.feature_map.parameters)[d]:x for d, x in enumerate(testdata)}), qargs=list(qr_xt))
        
        qc.h(qr_a)
        [qc.cswap(qr_a, qr_xi[d], qr_xt[d]) for d in range(len(qr_xi))]
        qc.h(qr_a)
        qc.measure(list(qr_a)+list(qr_yi), list(cr))
        return qc
    
    def _create_dual_circuit(self):
        qr_a = QuantumRegister(1, 'a')
        qr_i = QuantumRegister(int(math.log2(self.num_data)), 'i')
        qr_xi = QuantumRegister(int(math.log2(self.dim_data)), 'x')
        qr_yi = QuantumRegister(1, 'y')
        qr_j = QuantumRegister(len(qr_i), 'j')
        qr_xj = QuantumRegister(len(qr_xi), 'xj')
        qr_yj = QuantumRegister(len(qr_yi), 'yj')
        cr = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qr_a, qr_i, qr_xi, qr_yi, qr_j, qr_xj, qr_yj, cr)
        qc.compose(self.var_form, qubits=list(qr_i), inplace=True)
        qc.compose(self.var_form, qubits=list(qr_j), inplace=True)
        for ctrl_state in range(self.num_data):
            bind_feature_map = self.feature_map.bind_parameters({tuple(self.feature_map.parameters)[d]:x for d, x in enumerate(self.data[ctrl_state])})
            qc.append(bind_feature_map.to_gate().control(len(qr_i), ctrl_state=ctrl_state), qargs=list(qr_i)+list(qr_xi))
            qc.append(bind_feature_map.to_gate().control(len(qr_j), ctrl_state=ctrl_state), qargs=list(qr_j)+list(qr_xj))
            if self.label[ctrl_state]==1:
                qc.append(XGate().control(len(qr_i), ctrl_state=ctrl_state), qargs=list(qr_i)+list(qr_yi))
                qc.append(XGate().control(len(qr_j), ctrl_state=ctrl_state), qargs=list(qr_j)+list(qr_yj))
        
        qc.h(qr_a)
        [qc.cswap(qr_a, qr_xi[d], qr_xj[d]) for d in range(len(qr_xi))]
        qc.h(qr_a)
        qc.measure(list(qr_a)+list(qr_yi)+list(qr_yj), list(cr))
        return qc

    def evaluate_circuit(self, **kwargs):
        dict_primal = self.quantum_instance.execute(self.primal_qc, **kwargs).get_counts()
        dict_dual = self.quantum_instance.execute(self.dual_qc, **kwargs).result().get_counts()
        ret = dict()
        ret['ayk'] = self._process_counts(2, dict_primal, 0, 1)
        ret['ay'] = self._process_counts(2, dict_primal, 0)
        ret['aayyk'] = self._process_counts(3, dict_dual, 0, 1, 2)
        ret['aayy'] = self._process_counts(3, dict_dual, 1, 2)
        return ret

    def _run(self):
        return None

    def _process_counts(self, n, dic, *count):
        temp = 0
        for bin in product((0,1), repeat=n):
            val1 = (-1)**sum([bin[c] for c in count])
            val2 = dic.get(''.join(map(str, bin)), 0)
            temp += val1*val2
        return temp

    def _validate_datset(self, training_dataset: Optional[Dict[str, np.ndarray]]):
        self.data = training_dataset['data']
        self.label = training_dataset['label']
        self.num_data = self.data.shape[0]
        self.dim_data = self.data.shape[1]
        if self.num_data != self.label.size:
            raise AquaError('Not enough/More number of labels compare to dataset')

    def _construct_UD(self):
        qr_i = QuantumRegister(math.log2(self.num_data))
        qr_xi = QuantumRegister(math.log2(self.dim_data))
        qr_yi = QuantumRegister(1)
        qc = QuantumCircuit(qr_i, qr_xi, qr_yi)
        for ctrl_state in range(self.num_data):
            bind_feature_map = self.feature_map.bind_parameters({tuple(self.feature_map.parameters)[d]:x for d, x in enumerate(self.data[ctrl_state])})
            qc.append(bind_feature_map.to_gate().control(len(qr_i), ctrl_state=ctrl_state), qargs=list(qr_i)+list(qr_xi))
            if self.label[ctrl_state]==1:
                qc.append(XGate().control(len(qr_i), ctrl_state=ctrl_state), qargs=list(qr_i)+list(qr_yi))
        
        UnitarySimulator().run(qc)
        

class FeatureGate(ControlledGate):
    def __init__(self, feature_map:Union[QuantumCircuit, FeatureMap, Gate], num_ctrl_qubits:int):
        self.num_ctrl_qubits = num_ctrl_qubits
        if isinstance(feature_map, Gate):
            self.base_gate = feature_map
        elif isinstance(feature_map, QuantumCircuit):
            self.base_gate = feature_map.to_gate()
        else:
            raise NotImplementedError('FeatureMap(?)')
        
    def __call__(self, params:Union[List[float], np.ndarray], ctrl_state):
        if len(params) != len(self.base_gate.params):
            raise AquaError('Parameter Mismatch: Cannot Assign {:} to {:}'.format(params, self.base_gate.params))

class OneLocal(TwoLocal):
    def __init__(self, num_qubits:int=1, reps=1, rotation_blocks=['rx', 'rz'], **kwargs):
        super().__init__(num_qubits, reps=reps, rotation_blocks=rotation_blocks, skip_final_rotation_layer=True, **kwargs)



        
