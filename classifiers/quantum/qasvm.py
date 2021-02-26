
import logging
import numpy as np
import math
from itertools import product
from qiskit.aqua.aqua_error import AquaError
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQAlgorithm
from qiskit.aqua.algorithms.classifiers.vqc import VQAlgorithm
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.providers import Backend, BaseBackend
from sklearn.metrics import accuracy_score
from ..utils import postprocess_Z_expectation
from ._qasvm import QASVM
from .quantum_circuits import Bloch_uniform_QASVM_circuit, Bloch_sphere_QASVM_circuit, uniform_QASVM_circuit, _uc_QASVM_circuit
from typing import Union, Optional, Dict, List, Callable

logger = logging.getLogger(__name__)

class BlochSphereQASVM(QASVM):
    def __init__(self, 
                training_data:np.ndarray,
                training_label:np.ndarray,
                var_form: Union[QuantumCircuit, VariationalForm] = None,
                optimizer: Optimizer = None,
                gradient: Optional[Union[GradientBase, Callable]] = None,
                initial_point: Optional[np.ndarray] = None,
                quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                C:float = None, k:float = 0.1
                     ) -> None:
        theta = ParameterVector('X', 2)
        feature_map = QuantumCircuit(1)
        feature_map.u3(theta[0], theta[1], 0, 0)
        super().__init__(training_data=training_data, 
                        training_label=training_label, 
                        var_form=var_form, 
                        feature_map=feature_map, 
                        optimizer=optimizer, 
                        gradient=gradient, 
                        initial_point=initial_point, 
                        quantum_instance=quantum_instance, 
                        C=C, k=k)
        self.circuit_class = Bloch_sphere_QASVM_circuit


class UniformQASVM(QASVM):
    def __init__(self, 
            feature_map: Union[QuantumCircuit, FeatureMap],
            training_data:np.ndarray,
            training_label:np.ndarray,
            quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
            C:float = None, k:float = 0.1
                    ) -> None:
        super().__init__(training_data=training_data, training_label=training_label, feature_map=feature_map, quantum_instance=quantum_instance, C=C, k=k)
        self.circuit_class = uniform_QASVM_circuit

    def _run(self) -> Dict:
        test_x = ParameterVector('𝒳', len(self._feature_map_params))
        self.classifier_circuit = self._construct_first_order_circuit(test_x)
        self._classifier_circuit_parameters = test_x
        return dict()

class UniformBlochQASVM(BlochSphereQASVM):
    def __init__(self, 
            training_data:np.ndarray,
            training_label:np.ndarray,
            quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
            C:float = None, k:float = 0.1
                    ) -> None:
        super().__init__(training_data=training_data, training_label=training_label, quantum_instance=quantum_instance, C=C, k=k)
        self.circuit_class = Bloch_uniform_QASVM_circuit

    def _run(self) -> Dict:
        test_x = ParameterVector('𝒳', len(self._feature_map_params))
        self.classifier_circuit = self._construct_first_order_circuit(test_x)
        self._classifier_circuit_parameters = test_x
        return dict()
