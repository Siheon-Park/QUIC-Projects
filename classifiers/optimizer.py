from typing import Optional, List, Callable
import logging
import numpy as np
from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers.spsa import SPSA

from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

class MySPSA(SPSA):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA is an algorithmic method for optimizing systems with multiple unknown parameters.
    As an optimization method, it is appropriately suited to large-scale population models,
    adaptive modeling, and simulation optimization.

    .. seealso::
        Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.

    SPSA is a descent method capable of finding global minima,
    sharing this property with other methods as simulated annealing.
    Its main feature is the gradient approximation, which requires only two
    measurements of the objective function, regardless of the dimension of the optimization
    problem.

    .. note::

        SPSA can be used in the presence of noise, and it is therefore indicated in situations
        involving measurement uncertainty on a quantum computation when finding a minimum.
        If you are executing a variational algorithm using a Quantum ASseMbly Language (QASM)
        simulator or a real device, SPSA would be the most recommended choice among the optimizers
        provided here.

    The optimization process includes a calibration phase, which requires additional
    functional evaluations.

    For further details, please refer to https://arxiv.org/pdf/1704.05018v2.pdf#section*.11
    (Supplementary information Section IV.)
    """

    _C0 = 2 * np.pi * 0.1
    _OPTIONS = ['save_steps', 'last_avg']

    def __init__(self,
                 maxiter: int = 1000,
                 save_steps: int = 1,
                 last_avg: int = 1,
                 c0: float = _C0,
                 c1: float = 0.1,
                 c2: float = 0.602,
                 c3: float = 0.101,
                 c4: float = 0,
                 skip_calibration: bool = False,
                 callback:Callable = None,
                 max_trials: Optional[int] = None) -> None:
        """
        Args:
            maxiter: Maximum number of iterations to perform.
            save_steps: Save intermediate info every save_steps step. It has a min. value of 1.
            last_avg: Averaged parameters over the last_avg iterations.
                If last_avg = 1, only the last iteration is considered. It has a min. value of 1.
            c0: The initial a. Step size to update parameters.
            c1: The initial c. The step size used to approximate gradient.
            c2: The alpha in the paper, and it is used to adjust a (c0) at each iteration.
            c3: The gamma in the paper, and it is used to adjust c (c1) at each iteration.
            c4: The parameter used to control a as well.
            skip_calibration: Skip calibration and use provided c(s) as is.
            max_trials: Deprecated, use maxiter.
            callback(Callable): 
                callback Args : k, cost, theta, cost_plus, cost_minus, theta_plus, theta_minus
        """
        super().__init__(maxiter=maxiter,save_steps=save_steps,last_avg=last_avg,c0=c0,c1=c1,c2=c2,c3=c3,c4=c4,skip_calibration=skip_calibration,max_trials=max_trials)
        self.callback = callback

    def _optimization(self,
                      obj_fun: Callable,
                      initial_theta: np.ndarray,
                      maxiter: int,
                      save_steps: int = 1,
                      last_avg: int = 1) -> List:
        """Minimizes obj_fun(theta) with a simultaneous perturbation stochastic
        approximation algorithm.

        Args:
            obj_fun: the function to minimize
            initial_theta: initial value for the variables of obj_fun
            maxiter: the maximum number of trial steps ( = function
                calls/2) in the optimization
            save_steps: stores optimization outcomes each 'save_steps'
                trial steps
            last_avg: number of last updates of the variables to average
                on for the final obj_fun
        Returns:
            a list with the following elements:
                cost_final : final optimized value for obj_fun
                theta_best : final values of the variables corresponding to
                    cost_final
                cost_plus_save : array of stored values for obj_fun along the
                    optimization in the + direction
                cost_minus_save : array of stored values for obj_fun along the
                    optimization in the - direction
                theta_plus_save : array of stored variables of obj_fun along the
                    optimization in the + direction
                theta_minus_save : array of stored variables of obj_fun along the
                    optimization in the - direction
        """

        theta = initial_theta
        theta_best = np.zeros(initial_theta.shape)
        for k in tqdm(range(maxiter)):
            # SPSA Parameters
            a_spsa = float(self._parameters[0]) / np.power(k + 1 + self._parameters[4],
                                                           self._parameters[2])
            c_spsa = float(self._parameters[1]) / np.power(k + 1, self._parameters[3])
            delta = 2 * aqua_globals.random.integers(2, size=np.shape(initial_theta)[0]) - 1
            # plus and minus directions
            theta_plus = theta + c_spsa * delta
            theta_minus = theta - c_spsa * delta
            # cost function for the two directions
            if self._max_evals_grouped > 1:
                cost_plus, cost_minus = obj_fun(np.concatenate((theta_plus, theta_minus, theta)))
            else:
                cost_plus = obj_fun(theta_plus)
                cost_minus = obj_fun(theta_minus)
            if self.callback is not None:
                self.callback(k, obj_fun, theta, cost_plus, cost_minus, theta_plus, theta_minus)
            # derivative estimate
            g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
            # updated theta
            theta = theta - a_spsa * g_spsa
            # saving
            if k % save_steps == 0:
                logger.debug('Objective function at theta+ for step # %s: %1.7f', k, cost_plus)
                logger.debug('Objective function at theta- for step # %s: %1.7f', k, cost_minus)

            if k >= maxiter - last_avg:
                theta_best += theta / last_avg
        # final cost update
        cost_final = obj_fun(theta_best)
        logger.debug('Final objective function is: %.7f', cost_final)

        return [cost_final, theta_best, None, None, None, None]