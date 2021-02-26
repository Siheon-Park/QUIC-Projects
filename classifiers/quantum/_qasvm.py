
from abc import abstractmethod
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
from .quantum_circuits import QASVM_circuit
from typing import Union, Optional, Dict, List, Callable


class QASVM(VQAlgorithm):
    def __init__(self, 
                training_data:np.ndarray,
                training_label:np.ndarray,
                var_form: Union[QuantumCircuit, VariationalForm] = None,
                feature_map: Union[QuantumCircuit, FeatureMap] = None,
                optimizer: Optimizer = None,
                gradient: Optional[Union[GradientBase, Callable]] = None,
                initial_point: Optional[np.ndarray] = None,
                quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                C:float = 1, k:float = 0.1
                     ) -> None:
        super().__init__(var_form=var_form, optimizer=optimizer, quantum_instance=quantum_instance, cost_fn=None, gradient=gradient, initial_point=initial_point)
        self.circuit_class = QASVM_circuit
        self._data = training_data
        self._label = training_label
        self.num_data = self.data.shape[0]
        self.dim_data = self.data.shape[1]
        if self.num_data != self.label.size:
            raise AquaError('Not enough/More number of labels compare to dataset')
        self.feature_map = feature_map
        self.C = C
        self.k = k

        self._first_order_circuits = None
        self._second_order_circuit = None
        self._classifier_circuit = None
        self._classifier_circuit_parameters = None

        self.result = None

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def var_form(self) -> Optional[Union[QuantumCircuit, VariationalForm]]:
        """ Returns variational form """
        return self._var_form

    @property
    def feature_map(self) -> Optional[Union[QuantumCircuit, VariationalForm]]:
        """ Returns variational form """
        return self._feature_map

    @property
    def first_order_circuits(self):
        return self._first_order_circuits

    @property
    def second_order_circuit(self):
        return self._second_order_circuit

    @property
    def classifier_circuit(self):
        return self._classifier_circuit

    @property
    def cost_fn(self):
        return self._cost_fn

    @var_form.setter
    def var_form(self, var_form: Optional[Union[QuantumCircuit, VariationalForm]]):
        """ Sets variational form """
        if hasattr(var_form, 'ordered_parameters'):
            self._var_form_params = var_form.ordered_parameters
            self._var_form = var_form
        elif isinstance(var_form, QuantumCircuit):
            # store the parameters
            self._var_form_params = sorted(var_form.parameters, key=lambda x : int(x.name.split('[')[-1].split(']')[0]))
            self._var_form = var_form
        elif isinstance(var_form, VariationalForm):
            self._var_form_params = ParameterVector('θ', length=var_form.num_parameters)
            self._var_form = var_form
        elif var_form is None:
            self._var_form_params = None
            self._var_form = var_form
        else:
            raise ValueError('Unsupported type "{}" of var_form'.format(type(var_form)))

    @feature_map.setter
    def feature_map(self, feature_map: Optional[Union[QuantumCircuit, VariationalForm]]):
        """ Sets feature map """
        if hasattr(feature_map, 'ordered_paramters'):
            self._feature_map_params = self.ordered_parameters
            self._feature_map = feature_map
        elif isinstance(feature_map, QuantumCircuit):
            # store the parameters
            self._feature_map_params = sorted(feature_map.parameters, key=lambda x : int(x.name.split('[')[-1].split(']')[0]))
            self._feature_map = feature_map
        elif isinstance(feature_map, FeatureMap):
            self._feature_map_params = ParameterVector('θ', length=feature_map.num_parameters)
            self._feature_map = feature_map
        elif feature_map is None:
            self._feature_map_params = None
            self._feature_map = feature_map
        else:
            raise ValueError('Unsupported type "{}" of var_form'.format(type(feature_map)))

    @classifier_circuit.setter
    def classifier_circuit(self, circuit):
        self._classifier_circuit = circuit

    @cost_fn.setter
    def cost_fn(self, fn:Callable):
        self._cost_fn = fn

    @property
    def dual(self):
        """ dual mod """
        self.isdual = True
        self._second_order_circuit = self._construct_second_order_circuit()
        def cost_fn(param:np.ndarray):
            ret = self._evaluate_second_order_circuit(param)
            return ret['aayyk']+ret['aayy']/self.k
        self.cost_fn = cost_fn
        return self

    @property
    def primal(self):
        """ primal mode """
        self.isdual = False
        self._first_order_circuits = [self._construct_first_order_circuit(testdata) for testdata in self.data]
        self._second_order_circuit = self._construct_second_order_circuit()
        def cost_fn(param:np.ndarray):
            ret = self._evaluate_second_order_circuit(param)
            ret.update(self._evaluate_first_order_circuits(param))
            t = ret['ayk']+ret['ay']/self.k
            y = np.where(self.label>0, 1, -1)
            clf = np.sum(np.maximum(np.zeros_like(t), 1/self.C-y*t))
            reg = 0.5*(ret['aayyk']+ret['aayy']/self.k)
            return clf+reg
        self.cost_fn = cost_fn
        return self

    def _run(self) -> Dict:
        self.result = self.find_minimum(self.initial_point, self.var_form, self.cost_fn, self.optimizer, self._gradient)
        test_x = ParameterVector('𝒳', len(self._feature_map_params))
        self.classifier_circuit = self._construct_first_order_circuit(test_x).assign_parameters(self.result.optimal_parameters)
        self._classifier_circuit_parameters = test_x
        return dict(self.result)

    def get_optimal_cost(self):
        return self.result.optimal_value

    def get_optimal_circuit(self):
        return self.classifier_circuit

    def get_optimal_vector(self):
        return self.result.optimal_point

    @property
    def optimal_params(self):
        return self.result.optimal_parameters

    def _f(self, testdata:np.ndarray):
        ret = self._evaluate_classifier_circuit(testdata)
        return ret['ayk_x']+ret['ay_x']/self.k

    def f(self, testdata:np.ndarray):
        if len(testdata.shape)==1:
            return self._f(testdata)
        else:
            return np.array([self._f(t) for t in testdata])

    def predict(self, testdata:np.ndarray):
        return np.where(self.f(testdata)>0, 1., 0.)

    def accuracy(self, testdata:np.ndarray, testlabel:np.ndarray):
        return accuracy_score(self.predict(testdata), testlabel)

    def _construct_first_order_circuit(self, testdata:Union[np.ndarray, ParameterVector, List[Parameter]]):
        qc = self.circuit_class(self.num_data, self.dim_data, ord=1)
        qc.add_var_form(self.var_form)
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data)
        qc.X_encode(self.feature_map, self._feature_map_params, testdata=testdata)
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc
    
    def _construct_second_order_circuit(self):
        qc = self.circuit_class(self.num_data, self.dim_data, ord=2)
        qc.add_var_form(self.var_form, reg='i')
        qc.add_var_form(self.var_form, reg='j')
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data, reg='i')
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data, reg='j')
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc

    def _evaluate_second_order_circuit(self, param:Union[np.ndarray, List[float], List[np.ndarray]], **kwargs) -> Dict[str, float]:
        assert len(param)==len(self._var_form_params)
        param_dict = {p:v for p, v in zip(self._var_form_params, param)}
        if isinstance(self.quantum_instance, QuantumInstance):
            dict_dual = self.quantum_instance.execute(self.second_order_circuit.assign_parameters(param_dict)).get_counts()
        else: # BaseBackend, Backend
            dict_dual = execute(self.second_order_circuit.assign_parameters(param_dict), backend=self.quantum_instance, **kwargs).result().get_counts()
        eval_dict = dict()
        eval_dict['aayyk'] = postprocess_Z_expectation(3, dict_dual, 2, 1, 0)
        eval_dict['aayy'] = postprocess_Z_expectation(3, dict_dual, 1, 0)
        return eval_dict

    def _evaluate_first_order_circuits(self, param:List[float], **kwargs) -> Dict[str, float]:
        assert len(param)==len(self._var_form_params)
        param_dict = {p:v for p, v in zip(self._var_form_params, param)}
        if isinstance(self.quantum_instance, QuantumInstance):
            dict_primals = self.quantum_instance.execute([qc.assign_parameters(param_dict) for qc in self.first_order_circuits], **kwargs).get_counts()
        else: # BaseBackend, Backend
            dict_primals = execute([qc.assign_parameters(param_dict) for qc in self.first_order_circuits], backend=self.quantum_instance, **kwargs).result().get_counts()          
        eval_dict = dict()
        eval_dict['ayk'] = np.array([postprocess_Z_expectation(2, dict_primal, 1, 0) for dict_primal in dict_primals])
        eval_dict['ay'] = np.array([postprocess_Z_expectation(2, dict_primal, 0) for dict_primal in dict_primals])
        return eval_dict

    def _evaluate_classifier_circuit(self, param:List[float], **kwargs) -> Dict[str, float]:
        if self.classifier_circuit is not None:
            param_dict = {p:v for p, v in zip(self._classifier_circuit_parameters, param)}
        else:
            raise NotImplementedError('Run Algorithm first!')
        if isinstance(self.quantum_instance, QuantumInstance):
            dict_ = self.quantum_instance.execute(self.classifier_circuit.assign_parameters(param_dict=param_dict)).get_counts()
        else: # BaseBackend, Backend
            dict_ = execute(self.classifier_circuit.assign_parameters(param_dict=param_dict), backend=self.quantum_instance, **kwargs).result().get_counts()
        eval_dict = dict()
        eval_dict['ayk_x'] = postprocess_Z_expectation(2, dict_, 1, 0)
        eval_dict['ay_x'] = postprocess_Z_expectation(2, dict_, 0)
        return eval_dict