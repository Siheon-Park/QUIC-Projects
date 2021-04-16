from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, Optional, Union
import logging
import numpy as np
from qiskit.aqua import aqua_globals
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.algorithms.optimizers.spsa import SPSA as new_SPSA

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
    def __init__(self, model:object, 
                 c0: float = 2 * np.pi * 0.1,
                 c1: float = 0.1,
                 c2: float = 0.602,
                 c3: float = 0.101,
                 c4: float = 0) -> None:
        super().__init__(model, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4)
        self.k=0

    def calibrate()