from abc import ABC, abstractmethod
from typing import Callable
import logging
import numpy as np
from qiskit.aqua import aqua_globals
from qiskit.circuit.parametervector import ParameterVector

logger = logging.getLogger(__name__)

class StocasticOptimizer(ABC):
    def __init__(self, objective:Callable, params:ParameterVector, hyperparams:dict, initial_point:np.ndarray=None) -> None:
        super().__init__()
        self.objective = objective
        self.hyperparams = hyperparams
        self.initial_point=initial_point
        if self.initial_point is None:
            self.params = {p:None for p in params}
        else:
            self.params = dict(zip(params, initial_point))

    @abstractmethod
    def step(self):
        """ evolve optimizer """
        raise NotImplementedError

class SPSA(StocasticOptimizer):
    def __init__(self, objective:Callable, params:ParameterVector,
                 c0: float = 2 * np.pi * 0.1,
                 c1: float = 0.1,
                 c2: float = 0.602,
                 c3: float = 0.101,
                 c4: float = 0,
                 initial_point: np.ndarray = None) -> None:
        hyperparams = dict(c0=c0, c1=c1, c2=c2, c3=c3, c4=c4)
        if initial_point is None:
            initial_point = np.pi*(2*np.random.rand(len(params))-1)
        super().__init__(objective, params, hyperparams, initial_point)
        self.k=0

    def step(self, k:int=None):
        """ evolve SPSA """
        theta = np.array(list(self.params.values()))
        if k is not None:
            self.k = k
        # SPSA Parameters
        a_spsa = float(self.hyperparams['c0']) / np.power(self.k + 1 + self.hyperparams['c4'],
                                                        self.hyperparams['c2'])
        c_spsa = float(self.hyperparams['c1']) / np.power(self.k + 1, self.hyperparams['c3'])
        delta = 2 * aqua_globals.random.integers(2, size=np.shape(theta)[0]) - 1
        # plus and minus directions
        theta_plus = theta + c_spsa * delta
        theta_minus = theta - c_spsa * delta
        # cost function for the two directions
        cost_plus = self.objective(theta_plus)
        cost_minus = self.objective(theta_minus)
        # derivative estimate
        g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
        # updated theta
        theta = theta - a_spsa * g_spsa
        logger.debug('Objective function at theta+ for step # %s: %1.7f', self.k, cost_plus)
        logger.debug('Objective function at theta- for step # %s: %1.7f', self.k, cost_minus)
        for i, k in enumerate(self.params.keys()):
            self.params[k] = theta[i]
        self.k+=1

    def calibrate(self, maxiter:int=1000):
        """Calibrates and stores the SPSA parameters back.

        SPSA parameters are c0 through c5 stored in parameters array

        c0 on input is target_update and is the aimed update of variables on the first trial step.
        Following calibration c0 will be updated.

        c1 is initial_c and is first perturbation of initial_theta.
        """
        initial_theta = np.array(list(self.params.values()))
        num_steps_calibration = min(25, max(1, maxiter // 5))
        target_update = self.hyperparams['c0']
        initial_c = self.hyperparams['c1']
        delta_obj = 0
        logger.debug("Calibration...")
        for i in range(num_steps_calibration):
            if i % 5 == 0:
                logger.debug('calibration step # %s of %s', str(i), str(num_steps_calibration))
            delta = 2 * aqua_globals.random.integers(2, size=np.shape(initial_theta)[0]) - 1
            theta_plus = initial_theta + initial_c * delta
            theta_minus = initial_theta - initial_c * delta
            obj_plus = self.objective(theta_plus)
            obj_minus = self.objective(theta_minus)
            delta_obj += np.absolute(obj_plus - obj_minus) / num_steps_calibration

        # only calibrate if delta_obj is larger than 0
        if delta_obj > 0:
            self.hyperparams['c0'] = target_update * 2 / delta_obj \
                * self.hyperparams['c1'] * (self.hyperparams['c4'] + 1)
            logger.debug('delta_obj is 0, not calibrating (since this would set c0 to inf)')

        logger.debug('Calibrated SPSA parameter c0 is %.7f', self.hyperparams['c0'])