from abc import ABC, abstractmethod
from classifiers.quantum.qasvm import QASVM
from classifiers import Classifier
from typing import Callable, Dict, Iterator, Optional, Union
import logging
import numpy as np
from qiskit.aqua import aqua_globals
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.algorithms.optimizers.spsa import SPSA as new_SPSA
from qiskit.algorithms.optimizers.spsa import constant, powerseries, bernoulli_perturbation
from qiskit.algorithms.optimizers.spsa import CALLBACK

logger = logging.getLogger(__name__)

class SPSA(new_SPSA):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA [1] is an algorithmic method for optimizing systems with multiple unknown parameters.
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

    The optimization process can includes a calibration phase if neither the ``learning_rate`` nor
    ``perturbation`` is provided, which requires additional functional evaluations.
    (Note that either both or none must be set.) For further details on the automatic calibration,
    please refer to the supplementary information section IV. of [2].

    References:

        [1]: J. C. Spall (1998). An Overview of the Simultaneous Perturbation Method for Efficient
        Optimization, Johns Hopkins APL Technical Digest, 19(4), 482–492.
        `Online. <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

        [2]: A. Kandala et al. (2017). Hardware-efficient Variational Quantum Eigensolver for
        Small Molecules and Quantum Magnets. Nature 549, pages242–246(2017).
        `arXiv:1704.05018v2 <https://arxiv.org/pdf/1704.05018v2.pdf#section*.11>`_

    """    
    def __init__(self,
                 model: QASVM,
                 blocking: bool = False,
                 allowed_increase: Optional[float] = None,
                 trust_region: bool = False,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
                 resamplings: Union[int, Dict[int, int]] = 1,
                 perturbation_dims: Optional[int] = None
                 ) -> None:
        r"""
        Args:
            model: model to optimize. it should have attributes ``cost_fn(Callable)``, 
                ``parameters(Union[np.ndarray, Dict[Parameter, float], ParameterDict])``.
            blocking: If True, only accepts updates that improve the loss (minus some allowed
                increase, see next argument).
            allowed_increase: If blocking is True, this sets by how much the loss can increase
                and still be accepted. If None, calibrated automatically to be twice the
                standard deviation of the loss function.
            trust_region: If True, restricts norm of the random direction to be :math:`\leq 1`.
            learning_rate: A generator yielding learning rates for the parameter updates,
                :math:`a_k`. If set, also ``perturbation`` must be provided.
            perturbation: A generator yielding the perturbation magnitudes :math:`c_k`. If set,
                also ``learning_rate`` must be provided.
            resamplings: The number of times the gradient is sampled using a random direction to
                construct a gradient estimate. Per default the gradient is estimated using only
                one random direction. If an integer, all iterations use the same number of
                resamplings. If a dictionary, this is interpreted as
                ``{iteration: number of resamplings per iteration}``.
            perturbation_dims: The number of perturbed dimensions. Per default, all dimensions
                are perturbed, but a smaller, fixed number can be perturbed. If set, the perturbed
                dimensions are chosen uniformly at random.
        """
        try:
            self.model.cost_fn
        except AttributeError as e:
            raise AttributeError(e, "'cost_fn' should be Callable")
        try:
            self.model.parameters
        except ArithmeticError as e:
            raise AttributeError(e, "'parameters' should be Union[np.ndarray, Dict[Parameter, float], ParameterDict]")
        self.model = model
        super().__init__(blocking=blocking, allowed_increase=allowed_increase, trust_region=trust_region, learning_rate=learning_rate, perturbation=perturbation, perturbation_dims=perturbation_dims, resamplings=resamplings)
        del self.maxiter
        del self.last_avg

        self._prepare_optimization()


    def _prepare_optimization(self):
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        loss = self.model.cost_fn
        x = self.model.parameters
        if self.learning_rate is None and self.perturbation is None:
            get_learning_rate, get_perturbation = self.calibrate(loss, x)
            # get iterator
            eta = get_learning_rate()
            eps = get_perturbation()
        elif self.learning_rate is None or self.perturbation is None:
            raise ValueError('If one of learning rate or perturbation is set, both must be set.')
        else:
            # get iterator
            eta = self.learning_rate()
            eps = self.perturbation()

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            self.fx = loss(x)

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(loss, x)

        logger.info('=' * 30)
        logger.info('Starting SPSA optimization')

        self.eta = eta
        self.eps = eps
        self.k = 0

    def step(self, callback:CALLBACK=None, k:int=None):
        if k is None:
            k = self.k
        loss = self.model.cost_fn
        x = np.asarray(list(self.model.parameters.values()))
        # compute update
        update = self._compute_update(loss, x, k, next(self.eps))

        # trust region
        if self.trust_region:
            norm = np.linalg.norm(update)
            if norm > 1:  # stop from dividing by 0
                update = update / norm

        # compute next parameter value
        update = update * next(self.eta)
        x_next = x - update

        # blocking
        if self.blocking:
            fx_next = loss(x_next)

            self._nfev += 1
            if self.fx + self.allowed_increase <= fx_next:  # accept only if self.model.cost_fn improved
                if callback is not None:
                    callback(self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
                            False)  # not accepted

                logger.info('Iteration %s rejected.', k)
                return None
            self.fx = fx_next

        logger.info('Iteration %s done.', k)
        if callback is not None:
            callback(self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
                    True)  # accepted
        # update parameters
        self.model.parameters = x_next
        return None
