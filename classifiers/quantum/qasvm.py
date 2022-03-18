import logging
import numpy as np
from qiskit.circuit.parametervector import ParameterVector, Parameter
from qiskit.circuit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from sympy.logic.boolalg import Boolean
from . import postprocess_Z_expectation
from .quantum_circuits import QASVM_circuit, _CIRCUIT_CLASS_DICT
from . import QuantumError
from . import QuantumClassifier
from typing import Any, Union, Optional, Dict, List, Iterable

logger = logging.getLogger(__name__)


class QASVM(QuantumClassifier):
    """
        Quantum Approximated Support Vector Machine

        .. note::
            mutations: Dual, Primal, Class

        For further details, please refer to https://(future_arxiv address)

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
                 data: np.ndarray,
                 label: np.ndarray,
                 quantum_instance: QuantumInstance,
                 C: float = 1, k: float = 0.1, option: Union[str, Any] = 'Bloch_sphere',
                 var_form: Optional[QuantumCircuit] = None,
                 feature_map: Optional[QuantumCircuit] = None,
                 initial_point: Optional[np.ndarray] = None,
                 num_data_qubits=None, num_index_qubits=None
                 ) -> None:
        super().__init__(data, label)
        if num_data_qubits is None:
            self.num_data_qubits = int(np.log2(self.dim_data))
        else:
            self.num_data_qubits = num_data_qubits
        if num_index_qubits is None:
            self.num_index_qubits = int(np.log2(self.num_data))
        else:
            self.num_index_qubits = num_index_qubits
        # initials
        self.C = C
        self.k = k
        self._initialized = False
        self._quantum_instance = None
        self._circuit_class = None
        self._var_form = None  # set self._var_form_params, self.num_parameters, self.parameters
        self._var_form_params = None  # set from self.var_form
        self._num_parameters = None  # set from self.var_form
        self._initial_point = None  # self.num_parameters should be defined first
        self._parameters = None  # set from self.var_form
        self._feature_map = None  # set self._feature_map_params
        self._feature_map_params = None  # sef from self.feature_map
        self.naive_first_order_circuit = None  # self.var_form, self.feature_map, self.had_transpiled should be
        # defined first
        self.naive_second_order_circuit = None  # self.var_form, self.feature_map, self.had_transpiled should be
        # defined first
        self._had_transpiled = False  # self.naive... , self.quantum_instance should be defined first
        self.transpiled_first_order_circuit = None  # self.var_form, self.feature_map, self.had_transpiled should be
        # defined first
        self.transpiled_second_order_circuit = None  # self.var_form, self.feature_map, self.had_transpiled should be
        # defined first
        self._mode = None  # self.naive... self.transpiled... should be defined first
        self._alpha = None  # everything should have been defined first

        self.quantum_instance = quantum_instance  #
        self.circuit_class = option  #
        self.var_form = var_form  #
        self.initial_point = initial_point
        self.feature_map = feature_map  #
        self.initialized = True
        self.mode = None

    # self.initialized
    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, tf: Boolean):
        if self._initialized != tf:
            self._basic_circuits_construction()
            self._initialized = True

    # self.circuit_class
    @property
    def circuit_class(self):
        """ class to generate circuits. can be set from str ('QASVM', 'Bloch_sphere', 'uniform', 'Bloch_uniform',
        '_uc') """
        return self._circuit_class

    @circuit_class.setter
    def circuit_class(self, option: Union[QASVM_circuit, str]):
        if isinstance(option, str):
            try:
                option = _CIRCUIT_CLASS_DICT[option]
            except KeyError:
                raise QuantumError(
                    'No Class in {:}.py that corresponds to {:}'.format(QASVM_circuit.__module__, option))
        if not issubclass(option, QASVM_circuit):
            raise QuantumError('{:} is not subclass of {:}'.format(repr(option), QASVM_circuit.__name__))
        if self._circuit_class != option:
            self._circuit_class = option
            self.var_form = self.var_form
            self.feature_map = self.feature_map
            # self.initialized = False

    # self.initial_point
    @property
    def initial_point(self):
        """ starting point. Uniform[-pi, pi] if set to None """
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Union[None, np.ndarray]):
        if initial_point is None:
            self._initial_point = 2 * np.pi * (np.random.rand(self.num_parameters) - 1 / 2)
        else:
            self._initial_point = initial_point
        self.parameters = self.initial_point

    # var_form
    @property
    def var_form(self) -> Union[QuantumCircuit, None]:
        # noinspection SpellCheckingInspection
        """ variational form (Ansatz) of QASVM. parameters will change accordingly """
        return self._var_form

    @var_form.setter
    def var_form(self, var_form: Union[QuantumCircuit, None]):
        # TODO:
        if 'uniform' in self.circuit_class.__name__ and var_form is not None:
            logger.warning(
                "{:} ignores var_form. Set 'circuit_class' first to modify 'var_form'".format(self.circuit_class))
        else:
            if hasattr(var_form, 'ordered_parameters'):
                _var_form_params = var_form.ordered_parameters
            elif isinstance(var_form, QuantumCircuit):
                # store the parameters
                _var_form_params = sorted(var_form.parameters, key=lambda x: int(x.name.split('[')[-1].split(']')[0]))

            elif var_form is None:
                _var_form_params = []
            else:
                raise ValueError('Unsupported type "{:}" of var_form'.format(type(var_form)))
            self._var_form_params = {
                'i': ParameterVector('θ_i', len(_var_form_params)),
                'j': ParameterVector('θ_j', len(_var_form_params)),
                '0': ParameterVector('θ', len(_var_form_params))
            }
            self._var_form = var_form.assign_parameters(
                dict(zip(_var_form_params, self._var_form_params['0']))) if var_form is not None else var_form
            self._num_parameters = len(_var_form_params)
            # self._parameters = ParameterDict(zip(self._var_form_params['0'], np.empty(self._num_parameters)))
            self._parameters = ParameterArray(self._var_form_params['0'])
        self.initialized = False

    @property
    def parameters(self) -> np.ndarray:
        """ optimization parameters of QASVM. parameter values can change but not keys """
        return self._parameters

    '''
    @parameters.setter
    def parameters(self, params: Union[np.ndarray, Dict[Parameter, float]]):
        if isinstance(params, np.ndarray):
            for k, v in zip(self._parameters.keys(), params):
                self._parameters[k] = v
        elif isinstance(params, dict):
            for k in self._parameters.keys():
                self._parameters[k] = params[k]
        else:
            raise QuantumError
    '''

    @parameters.setter
    def parameters(self, params: np.ndarray):
        if len(self._parameters) != len(params):
            raise QuantumError(
                f'Expect np.ndarray of length {len(self._parameters)}, but received array of length {len(params)}')
        self._parameters.update(params)

    @property
    def num_parameters(self) -> int:
        """ number of parameters to be optimized """
        return self._num_parameters

    # feature_map
    @property
    def feature_map(self) -> Union[QuantumCircuit, None]:
        """ feature map (Data Encoding) of QASVM """
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: Union[QuantumCircuit, None]):
        # TODO:
        if 'Bloch' in self.circuit_class.__name__ and feature_map is not None:
            logger.warning(
                "{:} ignores feature_map. Set 'circuit_class' first to modify 'feature_map'".format(self.circuit_class))
        else:
            """ Sets feature map """
            if hasattr(feature_map, 'ordered_parameters'):
                _feature_map_params = feature_map.ordered_parameters
            elif isinstance(feature_map, QuantumCircuit):
                # store the parameters
                _feature_map_params = sorted(feature_map.parameters,
                                             key=lambda x: int(x.name.split('[')[-1].split(']')[0]))
            elif feature_map is None:
                _feature_map_params = ParameterVector('temp', 2)
            else:
                raise ValueError('Unsupported type "{:}" of feature_map'.format(type(feature_map)))
            self._feature_map_params = ParameterVector('X', len(_feature_map_params))
            self._feature_map = feature_map.assign_parameters(
                dict(zip(_feature_map_params, self._feature_map_params))) if feature_map is not None else feature_map
        self.initialized = False

    # quantum_instance
    @property
    def quantum_instance(self) -> QuantumInstance:
        """ quantum instance provided by qiskit.aqua for first and second order circuits to run on """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance):
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
    def had_transpiled(self, tf: bool):
        assert isinstance(tf, bool)

        if tf:
            if self.quantum_instance.compile_config['initial_layout'] is not None:
                _temp = self.quantum_instance.compile_config['initial_layout']
                self.quantum_instance.compile_config['initial_layout'] = _temp.first_dict
                self.transpiled_first_order_circuit = self.quantum_instance.transpile(self.naive_first_order_circuit)[0]
                self.quantum_instance.compile_config['initial_layout'] = _temp
                self.transpiled_second_order_circuit = self.quantum_instance.transpile(self.naive_second_order_circuit)[
                    0]
            else:
                self.transpiled_first_order_circuit = self.quantum_instance.transpile(self.naive_first_order_circuit)[0]
                self.transpiled_second_order_circuit = self.quantum_instance.transpile(self.naive_second_order_circuit)[
                    0]
        else:
            self.transpiled_first_order_circuit = None
            self.transpiled_second_order_circuit = None

        self._had_transpiled = tf

    # self.first_order_circuit
    @property
    def first_order_circuit(self):
        """ first order circuit of QASVM (naive/transpiled)"""
        if self.had_transpiled:
            return self.transpiled_first_order_circuit
        else:
            return self.naive_first_order_circuit

    # self.second_order_circuit
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
    def mode(self, mode: str):
        if mode not in ['Dual', 'Primal', 'Classifier']:
            self._mode = None
        else:
            self._mode = mode

    @property
    def cost_fn(self):
        """ cost function of QASVM according to mode """
        if self.mode == 'Dual':
            return self._cost_fn_second_order
        elif self.mode == 'Primal':
            return lambda x: self._cost_fn_second_order(x) / 2 + self._cost_fn_first_order(x)
        elif self.mode == 'Classifier':
            return self._cost_fn_first_order
        else:
            return None

    @property
    def grad_fn(self):
        """ gradient function of QASVM according to mode """
        if self.mode == 'Dual':
            return self._grad_fn_second_order
        elif self.mode == 'Primal':
            return None
        elif self.mode == 'Classifier':
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
    def alpha(self, params: np.ndarray):
        if self.var_form is None:
            return np.abs(params) / sum(np.abs(params))
        else:
            var_qc = self.var_form.assign_parameters(dict(zip(self.var_form.parameters, params)))
            var_qc.save_statevector()
            result = self.quantum_instance.execute(var_qc)
            return np.abs(result.get_statevector()) ** 2

    '''
    @property
    def alpha(self):
        """ corresponding weights analogous to that of SVM. Set it to None and try again for another circuit run """
        if self._alpha is None:
            qc = self.var_form.assign_parameters(self.parameters)
            qc.measure_all()
            prob_dict = self.quantum_instance.execute([qc], self.had_transpiled).get_counts()
            return np.array(
                [prob_dict.get(''.join(map(str, b)), 0) for b in product((0, 1), repeat=qc.num_qubits)]) / sum(
                prob_dict.values())
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, a: np.ndarray):
        self._alpha = a
    '''

    # methods
    # noinspection SpellCheckingInspection
    def _cost_fn_second_order(self, param: Union[List[float], np.ndarray, Dict[Parameter, float]]):
        ret = self._evaluate_second_order_circuit(param, param)
        reg = ret['aayyk'] + ret['aayy'] / self.k
        return reg

    def _cost_fn_first_order(self, param: Union[List[float], np.ndarray, Dict[Parameter, float]]):
        ret = self._evaluate_first_order_circuit(param, self.data)
        t = ret['ayk'] + ret['ay'] / self.k
        y = np.where(self.label > 0, 1, -1)
        #esty = np.where(t > 0, 1, -1)
        risk = np.mean(np.maximum(np.zeros_like(t), -y*t))
        # clf = np.sum(np.maximum(np.zeros_like(t), 1 / self.C - y * t))
        return risk

    # noinspection SpellCheckingInspection
    def _grad_fn_second_order(self, param: np.ndarray):
        shifts = np.pi / 2 * np.eye(len(param))
        gradients = []
        for shift in shifts:
            ret_plus = self._evaluate_second_order_circuit(param, param + shift)
            ret_minus = self._evaluate_second_order_circuit(param, param - shift)
            reg_plus = ret_plus['aayyk'] + ret_plus['aayy'] / self.k
            reg_minus = ret_minus['aayyk'] + ret_minus['aayy'] / self.k
            gradients.append(reg_plus - reg_minus)
        return np.array(gradients)

    def classifying_function(self, params: Union[np.ndarray, List[float], Dict[Parameter, float]],
                             input: Union[np.ndarray, List[np.ndarray]]) -> float:
        """ f(theta=params, X=input) 
            Args:
                params: information on optimization parameters
                input: information on test data
            Returns:
                classifying value
        """
        ret = self._evaluate_first_order_circuit(params, input)
        return ret['ayk'] + ret['ay'] / self.k

    # noinspection SpellCheckingInspection
    def f(self, testdata: Union[np.ndarray, List[np.ndarray]]):
        """ calculate optimal classifying value
            Args:
                testdata: it has to have more than one datum
                          else, wrap with list, i.e. [testdata]
            Return:
                f(np.ndarray): f(X)
        """
        return self.classifying_function(self.parameters, testdata)

    # ------------------------------ methods circuit constructions & evaluation ------------------------------------ #
    def _basic_circuits_construction(self):
        """ setting first and second order circuits and transpile if needed. """
        self.naive_second_order_circuit = self._construct_second_order_circuit()
        self.naive_first_order_circuit = self._construct_first_order_circuit()
        if 'ibmq_qasm_simulator' in self.quantum_instance.backend_name:
            self.had_transpiled = False
        elif self.quantum_instance.noise_config:
            self.had_transpiled = True
        elif 'ibmq' in self.quantum_instance.backend_name:
            self.had_transpiled = True
        else:
            self.had_transpiled = False

    def _construct_first_order_circuit(self):
        """ constructor of first-order-circuit where var_form parameters are 'theta', feature_map parameters are 'X'."""
        qc: QASVM_circuit = self.circuit_class(self.num_index_qubits, self.num_data_qubits, order=1)
        qc.add_var_form(self.var_form)
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label,
                     N=self.num_data)
        qc.X_encode(self.feature_map, self._feature_map_params, testdata=self._feature_map_params)
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc

    def _construct_second_order_circuit(self):
        """ constructor of second-order-circuit where var_form parameters are 'theta_i', 'theta_j'. No feature_map
        parameters """
        qc: QASVM_circuit = self.circuit_class(self.num_index_qubits, self.num_data_qubits, order=2)
        if self.var_form is not None:
            qc.add_var_form(
                self.var_form.assign_parameters(dict(zip(self._var_form_params['0'], self._var_form_params['i']))),
                reg='i')
            qc.add_var_form(
                self.var_form.assign_parameters(dict(zip(self._var_form_params['0'], self._var_form_params['j']))),
                reg='j')
        else:
            qc.add_var_form(self.var_form, reg='i')
            qc.add_var_form(self.var_form, reg='j')
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label,
                     N=self.num_data, reg='i')
        qc.UD_encode(self.feature_map, self._feature_map_params, training_data=self.data, training_label=self.label,
                     N=self.num_data, reg='j')
        qc.SWAP_test()
        qc.Z_expectation_measurement()
        return qc

    # noinspection SpellCheckingInspection
    def _evaluate_second_order_circuit(self, param1: Union[np.ndarray, List[float], dict],
                                       param2: Union[np.ndarray, List[float], dict]) -> Dict[str, float]:
        """ evaluating second-order-circuit.
            Args:
                param1: 'theta_i'
                param2: 'theta_j' 
            Return:
                Dict of Z evals (float)"""
        if isinstance(param1, dict):
            param_dict = dict(zip(self._var_form_params['i'], param1.values()))
        else:
            param_dict = dict(zip(self._var_form_params['i'], param1))
        if isinstance(param2, dict):
            param_dict.update(dict(zip(self._var_form_params['j'], param2.values())))
        else:
            param_dict.update(dict(zip(self._var_form_params['j'], param2)))
        _dict = self.quantum_instance.execute([self.second_order_circuit.assign_parameters(param_dict)],
                                              self.had_transpiled).get_counts()
        eval_dict = dict()
        eval_dict['aayyk'] = postprocess_Z_expectation(3, _dict, 2, 1, 0)
        eval_dict['aayy'] = postprocess_Z_expectation(3, _dict, 1, 0)
        return eval_dict

    def _evaluate_first_order_circuit(self, param: Union[np.ndarray, List[float], Dict[Parameter, float]],
                                      data: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, float]:
        """ evaluating first-order-circuit.
            Args:
                param: var_form parameters 'param'
                data: feature_map parameters 'X', but in the form of iterable data i.e. [datum]
            Return:
                Dict of Z evals (np.ndarray or float)."""
        if isinstance(param, dict):
            param_dict = param
        else:
            param_dict = dict(zip(self._var_form_params['0'], param))
        param_dict_list = [dict(zip(self._feature_map_params, datum)) for datum in data]
        [data_dict.update(param_dict) for data_dict in param_dict_list]
        qc_list = list(map(self.first_order_circuit.assign_parameters, param_dict_list))
        _dict = self.quantum_instance.execute(qc_list, self.had_transpiled).get_counts()
        eval_dict = dict()
        if isinstance(_dict, dict):
            eval_dict['ayk'] = postprocess_Z_expectation(2, _dict, 1, 0)
            eval_dict['ay'] = postprocess_Z_expectation(2, _dict, 0)
        else:  # isinstance(_dict, list):
            eval_dict['ayk'] = np.array([postprocess_Z_expectation(2, __dict, 1, 0) for __dict in _dict])
            eval_dict['ay'] = np.array([postprocess_Z_expectation(2, __dict, 0) for __dict in _dict])
        return eval_dict

    # ------------------------------------------------------ utils ---------------------------------------------------#
    def __repr__(self) -> str:
        string = [str(self), 'Circuit Class: {:}'.format(self.circuit_class.__name__),
                  'QuantumInstance: {:}'.format(str(self.quantum_instance))]
        return '\n'.join(string)

    def __str__(self) -> str:
        if self.mode is not None:
            _string = '{:}_QASVM (C={:}, k={:})'.format(self.mode, self.C, self.k)
        else:
            _string = 'QASVM mode is not set!! (C={:}, k={:})'.format(self.C, self.k)
        return _string


class NormQSVM(QASVM):
    def __init__(self,
                 data: np.ndarray,
                 label: np.ndarray,
                 quantum_instance: QuantumInstance,
                 var_form: Optional[QuantumCircuit],
                 feature_map: Optional[QuantumCircuit],
                 lamda: float = 1.0,
                 initial_point: Optional[np.ndarray] = None,
                 ) -> None:
        super().__init__(data, label, quantum_instance, k=lamda, option='NqSVM', var_form=var_form,
                         feature_map=feature_map, initial_point=initial_point, num_data_qubits=feature_map.num_qubits,
                         num_index_qubits=var_form.num_qubits)
        self.lamda = lamda
        self.mode = 'Dual'

    def __repr__(self):
        _string = [f'Normalized QSVM({self.num_parameters} number of params)', repr(self.var_form),
                   repr(self.feature_map)]
        return '\n'.join(_string)


class PseudoNormQSVM(QuantumClassifier):
    def __init__(self, data: np.ndarray, label: np.ndarray,
                 quantum_instance: QuantumInstance, lamda: float = 1.0,
                 feature_map: QuantumCircuit = None, var_form: QuantumCircuit = None,
                 initial_point: np.ndarray = None):
        super().__init__(data, label)
        del self.alpha
        self.polary = 2 * self.label - 1
        self.quantum_instance = quantum_instance
        self._qk = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance, enforce_psd=False)
        self.kernel_matrix = np.abs(self._qk.evaluate(self.data, self.data)) ** 2
        self.feature_map = feature_map
        self.var_form = var_form
        self.lamda = lamda

        if self.var_form is None:
            self._parameters = ParameterArray(ParameterVector('theta', self.num_data))
        else:
            self._parameters = ParameterArray(self.var_form.parameters)
        self.num_parameters = len(self.parameters)

        if initial_point is None:
            self.initial_point = np.pi * (2 * np.random.random(self.num_parameters) - 1)
        else:
            self.initial_point = initial_point
        self.parameters.update(self.initial_point)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, new_params):
        self._parameters.update(new_params)

    def alpha(self, params: np.ndarray):
        if self.var_form is None:
            return np.abs(params) / sum(np.abs(params))
        else:
            var_qc = self.var_form.assign_parameters(dict(zip(self.var_form.parameters, params)))
            var_qc.save_statevector()
            result = self.quantum_instance.execute(var_qc)
            return np.abs(result.get_statevector()) ** 2

    def cost_fn(self, params: np.ndarray):
        alpha = self.alpha(params)
        beta = alpha * self.polary
        K = self.kernel_matrix + (1 / self.lamda)
        ret = beta @ K @ beta.reshape(-1, 1)
        return ret.item()

    def f(self, testdata):
        beta = self.alpha(self.parameters) * self.polary
        K = np.abs(self._qk.evaluate(self.data, testdata)) ** 2
        K += 1 / self.lamda
        return beta @ K


'''
class ParameterDict(dict):
    def __add__(self, other):
        ret = ParameterDict()
        if isinstance(other, np.ndarray):
            for k, v in zip(self.keys(), other):
                ret[k] = self[k] + v
        elif isinstance(other, dict):
            for k, v in zip(self.keys(), other.keys()):
                ret[k] = self[k] + other.get(v, 0)
        elif isinstance(other, float) or isinstance(other, int):
            for k in self.keys():
                ret[k] = self[k] + other
        else:
            ret = self + other
        return ret

    def __sub__(self, other):
        ret = ParameterDict()
        if isinstance(other, np.ndarray):
            for k, v in zip(self.keys(), other):
                ret[k] = self[k] - v
        elif isinstance(other, dict):
            for k, v in zip(self.keys(), other.keys()):
                ret[k] = self[k] - other.get(v, 0)
        elif isinstance(other, float) or isinstance(other, int):
            for k in self.keys():
                ret[k] = self[k] - other
        else:
            ret = self - other
        return ret

    def from_dict(self, d: dict):
        for k, v in d.items():
            self[k] = v

    @property
    def size(self):
        return len(self)
'''


class ParameterArray(np.ndarray):
    def __new__(cls, parameter_vector: ParameterVector, *args, **kwargs):
        shape = len(parameter_vector)
        obj = super().__new__(cls, *args, shape=shape, **kwargs)
        obj.parameter_vector = parameter_vector
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.parameter_vector = getattr(obj, 'parameter_vector', None)

    def to_dict(self):
        return dict(zip(self.parameter_vector, self))

    def update(self, params: Union[List, np.ndarray, Iterable]):
        for i, p in enumerate(params):
            self[i] = p

    def __repr__(self):
        return self.to_dict().__repr__()
