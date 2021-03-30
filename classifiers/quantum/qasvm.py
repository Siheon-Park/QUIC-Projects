import logging
import numpy as np
from itertools import product
from numpy.lib.arraysetops import isin
from qiskit.aqua.algorithms.vq_algorithm import VQResult
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumCircuit
from qiskit.aqua import QuantumInstance
from . import postprocess_Z_expectation
from .quantum_circuits import QASVM_circuit, _CIRCUIT_CLASS_DICT
from . import QuantumError
from . import QuantumClassifier
from typing import Any, Union, Optional, Dict, List, Callable

logger = logging.getLogger(__name__)

class QASVM(QuantumClassifier):
    """
        Quantum Approximated Support Vector Machine

        .. note::
            mutations: Dual, Primal, Class

        For further details, please refer to https://
    """

    def __init__(self, 
                data:np.ndarray,
                label:np.ndarray,
                var_form: QuantumCircuit = None,
                feature_map: QuantumCircuit = None,
                quantum_instance: Optional[QuantumInstance] = None,
                initial_point: Optional[np.ndarray] = None,
                C:float = 1, k:float = 0.1, option:Union[str, Any]='Bloch_sphere', reps:int=1
                     ) -> None:
        super().__init__(data, label)
        if isinstance(option, str):
            try:
                option = _CIRCUIT_CLASS_DICT[option]
            except KeyError:
                raise QuantumError('No Class in {:}.py that corresponds to {:}'.format(QASVM_circuit.__module__, option))
        if not issubclass(option, QASVM_circuit):
            raise QuantumError('{:} is not subclass of {:}'.format(repr(option), QASVM_circuit.__name__))
        else:
            self.circuit_class = option
        self.var_form = var_form
        self.feature_map = feature_map
        self.quantum_instance = quantum_instance
        self.initial_point = initial_point
        self.C = C
        self.k = k

        self.naive_first_order_circuit = None
        self.naive_second_order_circuit = None

        self.transpiled_first_order_circuit = None
        self.transpiled_second_order_circuit = None
        self._had_transpiled = False
        self.__fuck_you__vs_code = None

        self.result = None
        self._alpha = None



# var_form
    @property
    def var_form(self) -> Optional[Union[QuantumCircuit]]:
        """ Returns variational form """
        return self._var_form

    @var_form.setter
    def var_form(self, var_form: Optional[Union[QuantumCircuit]]):
        """ Sets variational form """
        if 'uniform' in self.circuit_class.__name__ and var_form is not None:
            logger.warning("{:} ignores var_form. Setting it to None".format(self.circuit_class))
            var_form = None
        if hasattr(var_form, 'ordered_parameters'):
            _var_form_params = var_form.ordered_parameters
        elif isinstance(var_form, QuantumCircuit):
            # store the parameters
            _var_form_params = sorted(var_form.parameters, key=lambda x : int(x.name.split('[')[-1].split(']')[0]))

        elif var_form is None:
            _var_form_params = []
        else:
            raise ValueError('Unsupported type "{:}" of var_form'.format(type(var_form)))
        self._var_form_params = {
            'i': ParameterVector('θ_i', len(_var_form_params)), 
            'j': ParameterVector('θ_j', len(_var_form_params)),
            '0': ParameterVector('θ', len(_var_form_params))
            }
        self._var_form = var_form.assign_parameters(dict(zip(_var_form_params, self._var_form_params['0']))) if var_form is not None else var_form
        self.num_parameters = len(_var_form_params)

    @property
    def var_form_params(self):
        return self._var_form_params

    @property
    def optimization_params(self):
        return self.var_form_params['0']

# feature_map
    @property
    def feature_map(self) -> Optional[Union[QuantumCircuit]]:
        """ Returns feature_map """
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: Optional[Union[QuantumCircuit]]):
        if 'Bloch' in self.circuit_class.__name__ and feature_map is not None:
            logger.warning("{:} ignores feature_map. Setting it to None".format(self.circuit_class))
            feature_map = None
        """ Sets feature map """
        if hasattr(feature_map, 'ordered_paramters'):
            _feature_map_params = feature_map.ordered_parameters
        elif isinstance(feature_map, QuantumCircuit):
            # store the parameters
            _feature_map_params = sorted(feature_map.parameters, key=lambda x : int(x.name.split('[')[-1].split(']')[0]))
        elif feature_map is None:
            _feature_map_params = ParameterVector('temp', 2)
        else:
            raise ValueError('Unsupported type "{:}" of feature_map'.format(type(feature_map)))
        self._feature_map_params = ParameterVector('X', len(_feature_map_params))
        self._feature_map = feature_map.assign_parameters(dict(zip(_feature_map_params, self._feature_map_params))) if feature_map is not None else feature_map

    @property
    def feature_map_params(self):
        return self._feature_map_params

# quantum_instance
    @property
    def quantum_instance(self):
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance:QuantumInstance):
        # aqua bug
        self._quantum_instance = quantum_instance
        if quantum_instance.is_simulator & ~quantum_instance.is_local:
            try:
                self.quantum_instance.qjob_config['wait']
            except KeyError:
                pass
            else:
                logger.warning("'QuantumInstance.qjob_config' has key 'wait'. It will be deleted.")
                del self.quantum_instance.qjob_config['wait']

# had_transpiled
    @property
    def had_transpiled(self):
        return self._had_transpiled

    @had_transpiled.setter
    def had_transpiled(self, tf:bool):
        assert isinstance(tf, bool)

        if tf:
            self.transpiled_first_order_circuit = self.quantum_instance.transpile(self.naive_first_order_circuit)[0]
            self.transpiled_second_order_circuit = self.quantum_instance.transpile(self.naive_second_order_circuit)[0]
        else:
            self.transpiled_first_order_circuit = None
            self.transpiled_second_order_circuit = None

        self._had_transpiled = tf

#self.first_order_circuit
    @property
    def first_order_circuit(self):
        if self.had_transpiled:
            return self.transpiled_first_order_circuit
        else:
            return self.naive_first_order_circuit

#self.second_order_circuit
    @property
    def second_order_circuit(self):
        if self.had_transpiled:
            return self.transpiled_second_order_circuit
        else:
            return self.naive_second_order_circuit

# self.mutation
    @property
    def mutation(self):
        return self.__fuck_you__vs_code

    @mutation.setter
    def mutation(self, mutation:str):
        if mutation=='Dual':
            self.cost_fn = self.cost_fn_second_order
            self.grad_fn = self.grad_fn_second_order
            self.__fuck_you__vs_code = 'Dual'
        elif mutation=='Primal':
            self.cost_fn = lambda x:self.cost_fn_second_order(x)/2+self.cost_fn_first_order(x)
            self.grad_fn = None
            self.__fuck_you__vs_code = 'Primal'
        elif mutation=='Classifier':
            self.cost_fn = self.cost_fn_first_order
            self.grad_fn = None
            self.__fuck_you__vs_code = 'Classifier'
        else:
            self.cost_fn = None
            self.grad_fn = None
            self.__fuck_you__vs_code = None

    @property
    def dual(self):
        """ dual mod """
        self.basic_circuits_construction()
        self.mutation = 'Dual'
        return self

    @property
    def primal(self):
        """ primal mutation """
        self.basic_circuits_construction()
        self.mutation = 'Primal'
        return self

    @property
    def classifier(self):
        self.basic_circuits_construction()
        self.mutation = 'Classifier'
        return self

# self.alpha
    @property
    def alpha(self):
        if self.result is None:
            return self._alpha
        if self._alpha is None:
            qc = self.var_form.assign_parameters(self.optimal_params)
            qc.measure_all()
            prob_dict = self.quantum_instance.execute([qc], self.had_transpiled).get_counts()
            return np.array([prob_dict.get(''.join(map(str, bin)), 0) for bin in product((0, 1), repeat=qc.num_qubits)])/sum(prob_dict.values())
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, a:np.ndarray):
        self._alpha = a   

# methods
    def cost_fn_second_order(self, param:Union[List[float], np.ndarray]):
        ret = self._evaluate_second_order_circuit(param, param)
        reg = ret['aayyk']+ret['aayy']/self.k
        return reg

    def cost_fn_first_order(self, param:Union[List[float], np.ndarray]):
        ret = self._evaluate_first_order_circuit(param, self.data)
        t = ret['ayk']+ret['ay']/self.k
        y = np.where(self.label>0, 1, -1)
        clf = np.sum(np.maximum(np.zeros_like(t), 1/self.C-y*t))
        return clf

    def grad_fn_second_order(self, param:np.ndarray):
        shifts = np.pi/2*np.eye(len(param))
        gradients = []
        for shift in shifts:
            ret_plus = self._evaluate_second_order_circuit(param, param+shift)
            ret_minus = self._evaluate_second_order_circuit(param, param-shift)
            reg_plus = ret_plus['aayyk']+ret_plus['aayy']/self.k
            reg_minus = ret_minus['aayyk']+ret_minus['aayy']/self.k
            gradients.append(reg_plus-reg_minus)
        return np.array(gradients)

    def classifying_function(self, params:Union[np.ndarray, List[float]], input:Union[np.ndarray, List[np.ndarray]]):
        ret = self._evaluate_first_order_circuit(params, input)
        return ret['ayk']+ret['ay']/self.k

    def run(self, optimizer=None) -> Dict:
        logger.debug(repr(self))
        logger.info('running VQAlgorithm')
        self.result = self.run_optimizer(
            optimizer=optimizer, 
            parameters=self.optimization_params, 
            cost_fn=self.cost_fn, 
            initial_point=self.initial_point, 
            bounds= None if not hasattr(self.var_form, 'parameter_bounds') else self.var_form.parameter_bounds,
            gradient_fn=self.grad_fn)
        logger.debug(dict(self.result))
        return dict(self.result)

    def set_result(self, optimal_parameters:Union[np.ndarray, dict], opt_val:float=None, num_optimizer_evals:int=0, eval_time:float=0):
        opt_params_mapping = None
        if isinstance(optimal_parameters, dict):
            opt_params_mapping = optimal_parameters
            opt_params = np.array(list(optimal_parameters.values()))
        else:
            opt_params_mapping = dict(zip(self.optimization_params, optimal_parameters))
            opt_params = optimal_parameters
            
        result = VQResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val if opt_val is not None else self.cost_fn(opt_params)
        result.optimal_point = opt_params
        result.optimal_parameters = opt_params_mapping
        self.result = result
        return self.result

# ------------------------------------------------------ methods for after running ------------------------------------------------------ #
    def get_optimal_cost(self):
        return self.result.optimal_value

    def get_optimal_circuit(self):
        return self.first_order_circuit.assign_parameters(self.result.optimal_parameters)

    def get_optimal_vector(self):
        return self.result.optimal_point

    @property
    def optimal_params(self):
        return self.result.optimal_parameters

    def f(self, testdata:Union[np.ndarray, List[np.ndarray]]):
        """ calculate optimal classifying value
            Args:
                testdata: it has to have more than one datum
                          else, wrap with list, i.e. [testdata]
            Return:
                f(np.ndarray): f(X)
        """
        return self.classifying_function(self.optimal_params, testdata)

# ------------------------------------------------------ methods circuit constructions & evaluation ------------------------------------------------------ #
    def basic_circuits_construction(self):
        """ setting first and second order circuits and transpile if needed. """
        self.naive_second_order_circuit = self._construct_second_order_circuit()
        self.naive_first_order_circuit = self._construct_firt_order_circuit()
        if self.quantum_instance.is_local:
            self.had_transpiled = False
        else:
            self.had_transpiled = True

    def _construct_firt_order_circuit(self):
        """ constructor of first-order-circuit where var_form parameters are 'theta', feature_map parameters are 'X'."""
        qc = self.circuit_class(self.num_data, self.dim_data, ord=1)
        qc.add_var_form(self.var_form)
        qc.UD_encode(self.feature_map, self.feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data)
        qc.X_encode(self.feature_map, self.feature_map_params, testdata=self.feature_map_params)
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc
    
    def _construct_second_order_circuit(self):
        """ constructor of second-order-circuit where var_form parameters are 'theta_i', 'theta_j'. No feature_map parameters"""
        qc = self.circuit_class(self.num_data, self.dim_data, ord=2)
        if self.var_form is not None:
            qc.add_var_form(self.var_form.assign_parameters(dict(zip(self.optimization_params, self.var_form_params['i']))), reg='i')
            qc.add_var_form(self.var_form.assign_parameters(dict(zip(self.optimization_params, self.var_form_params['j']))), reg='j')
        else:
            qc.add_var_form(self.var_form, reg='i')
            qc.add_var_form(self.var_form, reg='j')
        qc.UD_encode(self.feature_map, self.feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data, reg='i')
        qc.UD_encode(self.feature_map, self.feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data, reg='j')
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc

    def _evaluate_second_order_circuit(self, param1:Union[np.ndarray, List[float], dict], param2:Union[np.ndarray, List[float], dict]) -> Dict[str, float]:
        """ evaluating second-order-circuit.
            Args:
                param1: 'theta_i'
                param2: 'theta_j' 
            Return:
                Dict of Z evals (float)"""
        if isinstance(param1, dict):
            param_dict = param1
        else:
            param_dict = dict(zip(self.var_form_params['i'], param1))
        if isinstance(param2, dict):
            param_dict.update(param2)
        else:
            param_dict.update(dict(zip(self._var_form_params['j'], param2)))
        _dict = self.quantum_instance.execute([self.second_order_circuit.assign_parameters(param_dict)], self.had_transpiled).get_counts()
        eval_dict = dict()
        eval_dict['aayyk'] = postprocess_Z_expectation(3, _dict, 2, 1, 0)
        eval_dict['aayy'] = postprocess_Z_expectation(3, _dict, 1, 0)
        return eval_dict

    def _evaluate_first_order_circuit(self, theta:Union[np.ndarray, List[float], dict], data:Union[np.ndarray, List[np.ndarray]]) -> Dict[str, float]:
        """ evaluating first-order-circuit.
            Args:
                theta: var_form parameters 'theta'
                data: feature_map parameters 'X', but in the form of iterable data i.e. [datum]
            Return:
                Dict of Z evals (np.ndarray or float)."""
        if isinstance(theta, dict):
            param_dict = theta
        else:
            param_dict = dict(zip(self.optimization_params, theta))
        param_dict_list = [dict(zip(self.feature_map_params, datum)) for datum in data]
        [data_dict.update(param_dict) for data_dict in param_dict_list]
        qc_list = list(map(self.first_order_circuit.assign_parameters, param_dict_list))
        _dict = self.quantum_instance.execute(qc_list, self.had_transpiled).get_counts()
        eval_dict = dict()
        if isinstance(_dict, dict):
            eval_dict['ayk'] = postprocess_Z_expectation(2, _dict, 1, 0)
            eval_dict['ay'] = postprocess_Z_expectation(2, _dict, 0)
        else: # isinstance(_dict, list):
            eval_dict['ayk'] = np.array([postprocess_Z_expectation(2, __dict, 1, 0) for __dict in _dict])
            eval_dict['ay'] = np.array([postprocess_Z_expectation(2, __dict, 0) for __dict in _dict])
        return eval_dict

 # ------------------------------------------------------ utils ---------------------------------------------------#
    def __repr__(self) -> str:
        string = []
        string.append(str(self))
        string.append('Circuit Class: {:}'.format(self.circuit_class.__name__))
        string.append('QuantumInstance: {:}'.format(str(self.quantum_instance)))
        return '\n'.join(string)

    def __str__(self) -> str:
        if self.mutation is not None:
            _string = '{:}_QASVM (C={:}, k={:})'.format(self.mutation, self.C, self.k)
        else:
            _string = 'QASVM mutation is not set!! (C={:}, k={:})'.format(self.C, self.k)
        return _string