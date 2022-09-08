import logging
import numpy as np
import math
from qiskit.circuit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from .. import QuantumClassifier
from ..quantum_circuits import NqSVM_circuit
from qiskit.providers.aer import StatevectorSimulator
from qiskit import transpile

import pennylane as qml
# from pennylane import numpy as np

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

STATEVECTOR_INSTANCE = QuantumInstance(backend = StatevectorSimulator())

class PseudoTensorSoftQASVM(QuantumClassifier):
    def __init__(
        self, data: np.ndarray, label: np.ndarray,
        device: qml.Device, 
        lamda: float = 1.0, C: float = None,
        feature_map: QuantumCircuit = None, var_form: qml.operation.Operation = None,
    ):
        super().__init__(data, label)
        del self.alpha
        self.polary = 2 * self.label - 1
        self.device = device
        if feature_map is None:
            self.kernel_matrix = data
        else:
            self._qk = QuantumKernel(feature_map=feature_map, quantum_instance=STATEVECTOR_INSTANCE, enforce_psd=False)
            self.kernel_matrix = self._qk.evaluate(self.data, self.data)
        self.feature_map = feature_map
        self.var_form = var_form
        self.lamda = lamda
        self.C = C

    def cost_fn(self, params: qml.numpy.tensor):
        @qml.qnode(device=self.device)
        def get_alpha(params):
            self.var_form(params)
            return qml.probs(self.device.wires)
        alpha = get_alpha(params)
        beta = alpha * self.polary
        if self.C is not None:
            K = self.kernel_matrix + (1 / self.lamda) + (qml.numpy.eye(self.num_data)/self.C)
        else:
            K = self.kernel_matrix + (1 / self.lamda)
        ret = qml.numpy.matmul(beta, qml.numpy.matmul(K, beta))
        return ret
        
    def f(self, testdata: np.ndarray, params: np.ndarray):
        @qml.qnode(device=self.device)
        def get_alpha(params):
            self.var_form(params)
            return qml.probs(self.device.wires)
        alpha = get_alpha(params)
        beta = alpha * self.polary
        if self.feature_map is None:
            K = testdata
        else:
            K = self._qk.evaluate(self.data, testdata)
        K += 1 / self.lamda
        return np.matmul(beta, K)

    def predict(self, testdata:np.ndarray, params:np.ndarray):
        fvec = self.f(testdata=testdata, params=params)
        return np.where(fvec>0, 1, 0)

    def accuracy(self, testdata: np.ndarray, testlabel: np.ndarray, params:np.ndarray):
        esty = self.predict(testdata=testdata, params=params)
        return accuracy_score(esty, testlabel)

class TensorSoftQASVM(QuantumClassifier):
    def __init__(
        self, data: np.ndarray, label: np.ndarray,
        device: qml.Device, 
        lamda: float = 1.0, C: float = None,
        feature_map: QuantumCircuit = None, var_form: qml.operation.Operation = None,
    ):
        super().__init__(data, label)
        del self.alpha
        self.device = device
        self.feature_map = feature_map
        self.var_form = var_form
        self.lamda = lamda
        self.C = C

    def _make_qasm_string_for_UD(self):
        qc = NqSVM_circuit(self.device.num_wires, self.feature_map.num_qubits, 2)
        qc.UD_encode(self.feature_map, feature_map_params=None, N=None, training_data=self.data, training_label=self.label, reg='i')
        qc.UD_encode(self.feature_map, feature_map_params=None, N=None, training_data=self.data, training_label=self.label, reg='j')
        qasm_string = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx']).qasm()
        return qasm_string

    def _make_qasm_string_for_Ux(self, testdata):
        qc = NqSVM_circuit(self.device.num_wires, self.feature_map.num_qubits, 1)
        qc.UD_encode(self.feature_map, feature_map_params=None, N=None, training_data=self.data, training_label=self.label, reg='i')
        qc.X_encode(self.feature_map, self.feature_map.parameters, testdata=testdata)
        qasm_string = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx']).qasm()
        return qasm_string
