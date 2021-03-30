from inspect import Parameter
import logging
import numpy as np
from itertools import product
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumCircuit
from qiskit.aqua import QuantumInstance
from sympy.logic.boolalg import Boolean
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

        For further details, please refer to https://(future_arxiv adress)

        Attributes
        ----------
            data : np.ndarray
                data to encode
            label : np.ndarray
                labels to encode {0, 1}
            quantum_instance : QuantumInstance
                quantum instance provided by qiskit.aqua for first and second order circuits to run on
            mode : Union[None, str]
                QASVM operation mode that determines cost_fn, grad_fn, and f
            cost_fn : Union[None, Callable]
                objective function of QASVM. set mode properly if None

    """

    def __init__(self, 
                data:np.ndarray,
                label:np.ndarray,
                quantum_instance: QuantumInstance,
                C:float = 1, k:float = 0.1, option:Union[str, Any]='Bloch_sphere',
                var_form: Optional[QuantumCircuit] = None,
                feature_map: Optional[QuantumCircuit] = None,
                initial_point: Optional[np.ndarray] = None
                     ) -> None:
        super().__init__(data, label)

        # initials
        self.C = C
        self.k = k
        self._initialized = False
        self._quantum_instance = None
        self._circuit_class = None
        self._var_form = None # set self._var_form_params, self.num_parameters, self.parameters
        self._var_form_params = None # set from self.var_form
        self._num_parameters = None # set from self.var_form
        self._initial_point = None # self.num_parameters should be defined first
        self._parameters = {} # set from self.var_form
        self._feature_map = None # set self._feature_map_params
        self._feature_map_params = None # sef from self.feature_map
        self.naive_first_order_circuit = None # self.var_form, self.feature_map, self.had_transpiled should be defined first
        self.naive_second_order_circuit = None # self.var_form, self.feature_map, self.had_transpiled should be defined first
        self._had_transpiled = False # self.naive... , self.quantum_instance should be defined first
        self.transpiled_first_order_circuit = None # self.var_form, self.feature_map, self.had_transpiled should be defined first
        self.transpiled_second_order_circuit = None # self.var_form, self.feature_map, self.had_transpiled should be defined first
        self._mode = None # self.naive... self.transpiled... should be defined first
        self._alpha = None # everything should have been defined first

        self.circuit_class = option #
        self.quantum_instance = quantum_instance #
        self.var_form = var_form #
        self.initial_point = initial_point
        self.feature_map = feature_map #
        self.initialized = True
        self.mode = None

# self.initialized
    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, tf:Boolean):
        if self._initialized != tf:
            self._basic_circuits_construction()
            self._initialized = True

# self.circuit_class
    @property
    def circuit_class(self):
        """ class to generate circuits. can be set from str ('QASVM', 'Bloch_sphere', 'uniform', 'Bloch_uniform', '_uc') """
        return self._circuit_class

    @circuit_class.setter
    def circuit_class(self, option:Union[QASVM_circuit, str]):
        if isinstance(option, str):
            try:
                option = _CIRCUIT_CLASS_DICT[option]
            except KeyError:
                raise QuantumError('No Class in {:}.py that corresponds to {:}'.format(QASVM_circuit.__module__, option))
        if not issubclass(option, QASVM_circuit):
            raise QuantumError('{:} is not subclass of {:}'.format(repr(option), QASVM_circuit.__name__))
        if self._circuit_class != option:
            self._circuit_class = option
            self.initialized = False

# self.initial_point
    @property
    def initial_point(self):
        """ starting point. Uniform[-pi, pi] if set to None """
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point:Union[None, np.ndarray]):
        if initial_point is None:
            self._initial_point = 2*np.pi*(np.random.rand(self.num_parameters)-1/2)
        else:
            self._initial_point = initial_point
        self.parameters = self.initial_point

# var_form
    @property
    def var_form(self) -> Union[QuantumCircuit, None]:
        """ variational form (Ansatz) of QASVM. parameters will change accordingly """
        return self._var_form

    @var_form.setter
    def var_form(self, var_form: Union[QuantumCircuit, None]):
        """ Attributs:
                _var_form_params: Dict[Parameter, float]
                initial_point: 
        """
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
        self._num_parameters = len(_var_form_params)
        self._parameters = dict(zip(self._var_form_params['0'], np.empty(self._num_parameters)))

    @property
    def parameters(self)->Dict[Parameter, float]:
        """ optimization parameters of QASVM. parameter values can change but not keys """
        return self._parameters

    @parameters.setter
    def parameters(self, params:np.ndarray):
        for k, v in zip(self._parameters.keys(), params):
            self._parameters[k] = v

    @property
    def num_parameters(self)->int:
        """ number of parameters to be optimized """
        return self._num_parameters

# feature_map
    @property
    def feature_map(self) -> Union[QuantumCircuit, None]:
        """ feature map (Data Encoding) of QASVM """
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: Union[QuantumCircuit, None]):
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

# quantum_instance
    @property
    def quantum_instance(self)->QuantumInstance:
        """ quantum instance provided by qiskit.aqua for first and second order circuits to run on """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance:QuantumInstance):
        self._quantum_instance = quantum_instance
        self.initialized = False

# had_transpiled
    @property
    def had_transpiled(self):
        """ 
            Boolean variable to determine if the circuits had been transpiled.
            first and second order circuits will change accordingly
        """
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
        """ first order circuit of QASVM (naive/transpiled)"""
        if self.had_transpiled:
            return self.transpiled_first_order_circuit
        else:
            return self.naive_first_order_circuit

#self.second_order_circuit
    @property
    def second_order_circuit(self):
        """ second order circuit of QASVM (naive/transpiled)"""
        if self.had_transpiled:
            return self.transpiled_second_order_circuit
        else:
            return self.naive_second_order_circuit

# self.mode
    @property
    def mode(self):
        """ 
            str: operation mode of QASVM. 
            there are three options; ''Dual'', ''Primal'', 'Classifier'. 
            other options will reset mode to None.
        """
        return self._mode

    @mode.setter
    def mode(self, mode:str):
        if mode not in ['Dual', 'Primal', 'Classifier']:
            self._mode = None
        else:
            self._mode = mode

    @property
    def cost_fn(self):
        """ cost function of QASVM according to mode """
        if self.mode=='Dual':
            return self._cost_fn_second_order
        elif self.mode=='Primal':
            return lambda x:self._cost_fn_second_order(x)/2+self._cost_fn_first_order(x)
        elif self.mode=='Classifier':
            return self._cost_fn_first_order
        else:
            return None

    @property
    def grad_fn(self):
        """ gradient function of QASVM according to mode """
        if self.mode=='Dual':
            return self._grad_fn_second_order
        elif self.mode=='Primal':
            return None
        elif self.mode=='Classifier':
            return None
        else:
            return None
            
    @property
    def dual(self):
        """ set QASVM to dual mode and return self """
        self.mode = 'Dual'
        return self

    @property
    def primal(self):
        """  set QASVM to primal mode and return self """
        self.mode = 'Primal'
        return self

    @property
    def classifier(self):
        """ set QASVM to classifier mode and return self """
        self.mode = 'Classifier'
        return self

# self.alpha
    @property
    def alpha(self):
        """ corresponding weights analogos to that of SVM. Set it to None and try again for another circuit run """ 
        if self._alpha is None:
            qc = self.var_form.assign_parameters(self.parameters)
            qc.measure_all()
            prob_dict = self.quantum_instance.execute([qc], self.had_transpiled).get_counts()
            return np.array([prob_dict.get(''.join(map(str, bin)), 0) for bin in product((0, 1), repeat=qc.num_qubits)])/sum(prob_dict.values())
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, a:np.ndarray):
        self._alpha = a   

# methods
    def _cost_fn_second_order(self, param:Union[List[float], np.ndarray, Dict[Parameter, float]]):
        ret = self._evaluate_second_order_circuit(param, param)
        reg = ret['aayyk']+ret['aayy']/self.k
        return reg

    def _cost_fn_first_order(self, param:Union[List[float], np.ndarray, Dict[Parameter, float]]):
        ret = self._evaluate_first_order_circuit(param, self.data)
        t = ret['ayk']+ret['ay']/self.k
        y = np.where(self.label>0, 1, -1)
        clf = np.sum(np.maximum(np.zeros_like(t), 1/self.C-y*t))
        return clf

    def _grad_fn_second_order(self, param:np.ndarray):
        shifts = np.pi/2*np.eye(len(param))
        gradients = []
        for shift in shifts:
            ret_plus = self._evaluate_second_order_circuit(param, param+shift)
            ret_minus = self._evaluate_second_order_circuit(param, param-shift)
            reg_plus = ret_plus['aayyk']+ret_plus['aayy']/self.k
            reg_minus = ret_minus['aayyk']+ret_minus['aayy']/self.k
            gradients.append(reg_plus-reg_minus)
        return np.array(gradients)

    def classifying_function(self, params:Union[np.ndarray, List[float], Dict[Parameter, float]], input:Union[np.ndarray, List[np.ndarray]])->float:
        """ f(theta=params, X=input) 
            Args:
                params: information on optimization parameters
                input: information on test data
            Returns:
                classifying value
        """
        ret = self._evaluate_first_order_circuit(params, input)
        return ret['ayk']+ret['ay']/self.k

    def f(self, testdata:Union[np.ndarray, List[np.ndarray]]):
        """ calculate optimal classifying value
            Args:
                testdata: it has to have more than one datum
                          else, wrap with list, i.e. [testdata]
            Return:
                f(np.ndarray): f(X)
        """
        return self.classifying_function(self.parameters, testdata)

# ------------------------------------------------------ methods circuit constructions & evaluation ------------------------------------------------------ #
    def _basic_circuits_construction(self):
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
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data)
        qc.X_encode(self.feature_map, self._feature_map_params, testdata=self._feature_map_params)
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc
    
    def _construct_second_order_circuit(self):
        """ constructor of second-order-circuit where var_form parameters are 'theta_i', 'theta_j'. No feature_map parameters"""
        qc = self.circuit_class(self.num_data, self.dim_data, ord=2)
        if self.var_form is not None:
            qc.add_var_form(self.var_form.assign_parameters(dict(zip(self._var_form_params['0'], self._var_form_params['i']))), reg='i')
            qc.add_var_form(self.var_form.assign_parameters(dict(zip(self._var_form_params['0'], self._var_form_params['j']))), reg='j')
        else:
            qc.add_var_form(self.var_form, reg='i')
            qc.add_var_form(self.var_form, reg='j')
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data, reg='i')
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label, N=self.num_data, reg='j')
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
            param_dict = dict(zip(self._var_form_params['i'], param1))
        if isinstance(param2, dict):
            param_dict.update(param2)
        else:
            param_dict.update(dict(zip(self._var_form_params['j'], param2)))
        _dict = self.quantum_instance.execute([self.second_order_circuit.assign_parameters(param_dict)], self.had_transpiled).get_counts()
        eval_dict = dict()
        eval_dict['aayyk'] = postprocess_Z_expectation(3, _dict, 2, 1, 0)
        eval_dict['aayy'] = postprocess_Z_expectation(3, _dict, 1, 0)
        return eval_dict

    def _evaluate_first_order_circuit(self, theta:Union[np.ndarray, List[float], Dict[Parameter, float]], data:Union[np.ndarray, List[np.ndarray]]) -> Dict[str, float]:
        """ evaluating first-order-circuit.
            Args:
                theta: var_form parameters 'theta'
                data: feature_map parameters 'X', but in the form of iterable data i.e. [datum]
            Return:
                Dict of Z evals (np.ndarray or float)."""
        if isinstance(theta, dict):
            param_dict = theta
        else:
            param_dict = dict(zip(self._var_form_params['0'], theta))
        param_dict_list = [dict(zip(self._feature_map_params, datum)) for datum in data]
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
        if self.mode is not None:
            _string = '{:}_QASVM (C={:}, k={:})'.format(self.mode, self.C, self.k)
        else:
            _string = 'QASVM mode is not set!! (C={:}, k={:})'.format(self.C, self.k)
        return _string