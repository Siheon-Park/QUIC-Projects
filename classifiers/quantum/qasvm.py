import logging
import numpy as np
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit import execute
from qiskit.circuit import QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQAlgorithm, VQResult
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.providers import Backend, BaseBackend
from sklearn.metrics import accuracy_score
from . import postprocess_Z_expectation
from .quantum_circuits import QASVM_circuit, _CIRCUIT_CLASS_DICT
from . import QuantumError
from .. import Classifier
from ..datasets.__init__ import DatasetError
from typing import Any, Union, Optional, Dict, List, Callable

logger = logging.getLogger(__name__)

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
                C:float = 1, k:float = 0.1, option:Union[str, Any]='Bloch_sphere', reps:int=1
                     ) -> None:
        super().__init__(var_form=var_form, optimizer=optimizer, quantum_instance=quantum_instance, cost_fn=None, gradient=gradient, initial_point=initial_point)
        if isinstance(option, str):
            try:
                option = _CIRCUIT_CLASS_DICT[option]
            except KeyError:
                raise QuantumError('No Class in {:}.py that corresponds to {:}'.format(QASVM_circuit.__module__, option))
        if not issubclass(option, QASVM_circuit):
            raise QuantumError('{:} is not subclass of {:}'.format(repr(option), QASVM_circuit.__name__))
        else:
            self.circuit_class = option
        self._data = training_data
        self._label = training_label
        self.num_data = self.data.shape[0]
        self.dim_data = self.data.shape[1]
        if self.num_data != self.label.size:
            raise DatasetError('Not enough/More number of labels compare to dataset')
        self.feature_map = feature_map
        self.var_form = var_form
        self.C = C
        self.k = k

        self.naive_first_order_circuits = None
        self.naive_second_order_circuit = None
        self.naive_classifier_circuit = None
        self.classifier_circuit_parameters = None

        self.transpiled_first_order_circuits = None
        self.transpiled_second_order_circuit = None
        self.transpiled_classifier_circuit = None
        self._had_transpiled = False

        self.result = None

        # aqua bug
        if isinstance(self.quantum_instance, QuantumInstance):
            try:
                self.quantum_instance.qjob_config['wait']
            except KeyError:
                pass
            else:
                if quantum_instance.is_simulator:
                    logging.debug("'QuantumInstance.qjob_config' has key 'wait'. It will be deleted.")
                    del self.quantum_instance.qjob_config['wait']

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
        """ Returns feature_map """
        return self._feature_map

    @property
    def had_transpiled(self):
        return self._had_transpiled

    @property
    def first_order_circuits(self):
        if self.had_transpiled:
            return self.transpiled_first_order_circuits
        else:
            return self.naive_first_order_circuits

    @property
    def second_order_circuit(self):
        if self.had_transpiled:
            return self.transpiled_second_order_circuit
        else:
            return self.naive_second_order_circuit

    @property
    def classifier_circuit(self):
        if self.had_transpiled:
            return self.transpiled_classifier_circuit
        else:
            return self.naive_classifier_circuit

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
            self._var_form_params = []
            self._var_form = var_form
        else:
            raise ValueError('Unsupported type "{:}" of var_form'.format(type(var_form)))

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
            self._feature_map_params = ParameterVector('X', length=self.data.shape[1])
            self._feature_map = feature_map
        else:
            raise ValueError('Unsupported type "{:}" of var_form'.format(type(feature_map)))

    @property
    def dual(self):
        """ dual mod """
        self._setting('dual')
        return self

    @property
    def primal(self):
        """ primal mode """
        self._setting('primal')
        return self

    def _setting(self, option):
        if option not in ['dual', 'primal']:
            option = 'dual'
        self.classifier_circuit_parameters = ParameterVector('𝒳', len(self._feature_map_params))
        if option == 'dual':
            self.isdual = True
            self.naive_second_order_circuit = [self._construct_second_order_circuit()]
            self.naive_classifier_circuit = [self._construct_first_order_circuit(self.classifier_circuit_parameters)]
            if not self.quantum_instance.is_local:
                self.transpiled_second_order_circuit = self.quantum_instance.transpile(self.naive_second_order_circuit)
                self.transpiled_classifier_circuit = self.quantum_instance.transpile(self.naive_classifier_circuit)
                self._had_transpiled = True
            def cost_fn(param:np.ndarray):
                ret = self._evaluate_second_order_circuit(param)
                reg = ret['aayyk']+ret['aayy']/self.k
                return reg
        else: # option == 'primal'
            self.isdual = False
            self.naive_first_order_circuits = [self._construct_first_order_circuit(testdata) for testdata in self.data]
            self.naive_second_order_circuit = [self._construct_second_order_circuit()]
            self.naive_classifier_circuit = [self._construct_first_order_circuit(self.classifier_circuit_parameters)]
            if not self.quantum_instance.is_local:
                self.transpiled_second_order_circuit = self.quantum_instance.transpile(self.naive_second_order_circuit)
                self.transpiled_classifier_circuit = self.quantum_instance.transpile(self.naive_classifier_circuit)
                self.transpiled_first_order_circuits = [self.transpiled_classifier_circuit[0].assign_parameters({p:v for p,v in zip(self.classifier_circuit_parameters, testdata)}) for testdata in self.data]
                self._had_transpiled = True

            def cost_fn(param:np.ndarray):
                ret = self._evaluate_second_order_circuit(param)
                ret.update(self._evaluate_first_order_circuits(param))
                t = ret['ayk']+ret['ay']/self.k
                y = np.where(self.label>0, 1, -1)
                clf = np.sum(np.maximum(np.zeros_like(t), 1/self.C-y*t))
                reg = 0.5*(ret['aayyk']+ret['aayy']/self.k)
                return clf+reg
            
        self.cost_fn = cost_fn

    def _run(self) -> Dict:
        logger.debug(repr(self))
        if self.var_form is not None:
            logger.info('running VQAlgorithm')
            self.result = self.find_minimum(self.initial_point, self.var_form, self.cost_fn, self.optimizer, self._gradient)
            logger.debug(dict(self.result))
        else:
            logger.info('Cannot run VQAlogrithm since there is no variational form')
            result = VQResult()
            result.optimizer_evals = 0
            result.optimizer_time = 0
            result.optimal_value = self.cost_fn([])
            result.optimal_point = []
            result.optimal_parameters = {}
            self.result = result
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

    def __repr__(self) -> str:
        string = []
        string.append(str(self))
        string.append('Circuit Class: {:}'.format(self.circuit_class.__name__))
        string.append('QuantumInstance: {:}'.format(str(self.quantum_instance)))
        string.append(str(self.optimizer.setting))
        return '\n'.join(string)

    def __str__(self) -> str:
        return '{:}_QASVM (C={:}, k={:})'.format('Dual' if self.isdual else 'Primal', self.C, self.k)

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
            dict_dual = self.quantum_instance.execute([self.second_order_circuit[0].assign_parameters(param_dict)], self.had_transpiled).get_counts()
        else: # BaseBackend, Backend
            dict_dual = execute([self.second_order_circuit[0].assign_parameters(param_dict)], backend=self.quantum_instance, **kwargs).result().get_counts()
        eval_dict = dict()
        eval_dict['aayyk'] = postprocess_Z_expectation(3, dict_dual, 2, 1, 0)
        eval_dict['aayy'] = postprocess_Z_expectation(3, dict_dual, 1, 0)
        return eval_dict

    def _evaluate_first_order_circuits(self, param:List[float], **kwargs) -> Dict[str, float]:
        assert len(param)==len(self._var_form_params)
        param_dict = {p:v for p, v in zip(self._var_form_params, param)}
        if isinstance(self.quantum_instance, QuantumInstance):
            dict_primals = self.quantum_instance.execute([qc.assign_parameters(param_dict) for qc in self.first_order_circuits], self.had_transpiled).get_counts()
        else: # BaseBackend, Backend
            dict_primals = execute([qc.assign_parameters(param_dict) for qc in self.first_order_circuits], backend=self.quantum_instance, **kwargs).result().get_counts()          
        eval_dict = dict()
        eval_dict['ayk'] = np.array([postprocess_Z_expectation(2, dict_primal, 1, 0) for dict_primal in dict_primals])
        eval_dict['ay'] = np.array([postprocess_Z_expectation(2, dict_primal, 0) for dict_primal in dict_primals])
        return eval_dict

    def _evaluate_classifier_circuit(self, param:List[float], **kwargs) -> Dict[str, float]:
        if self.result is not None:
            param_dict = {p:v for p, v in zip(self.classifier_circuit_parameters, param)}
            param_dict.update(self.result.optimal_parameters)
        else:
            raise NotImplementedError('Run Algorithm first!')
        if isinstance(self.quantum_instance, QuantumInstance):
            dict_ = self.quantum_instance.execute([self.classifier_circuit[0].assign_parameters(param_dict=param_dict)], self.had_transpiled).get_counts()
        else: # BaseBackend, Backend
            dict_ = execute([self.classifier_circuit[0].assign_parameters(param_dict=param_dict)], backend=self.quantum_instance, **kwargs).result().get_counts()
        eval_dict = dict()
        eval_dict['ayk_x'] = postprocess_Z_expectation(2, dict_, 1, 0)
        eval_dict['ay_x'] = postprocess_Z_expectation(2, dict_, 0)
        return eval_dict

QASVM.save = Classifier.save