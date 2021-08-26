from time import time

from classifiers.quantum.qasvm import QASVM
from typing import Callable, Dict, Iterator, Optional, Union
import logging
import numpy as np
from qiskit.algorithms.optimizers.spsa import SPSA
from qiskit.algorithms.optimizers.spsa import CALLBACK, _validate_pert_and_learningrate

logger = logging.getLogger(__name__)


class tSPSA(SPSA):
    def __init__(self, *args, **kwargs):
        super(tSPSA, self).__init__(*args, **kwargs)
        self.num_step = 0

    def step(self, loss, initial_point):
        if self.num_step == 0:
            if self.learning_rate is None and self.perturbation is None:
                get_eta, get_eps = self.calibrate(loss, initial_point)
            else:
                get_eta, get_eps = _validate_pert_and_learningrate(
                    self.perturbation, self.learning_rate
                )
            eta, eps = get_eta(), get_eps()

            if self.lse_solver is None:
                lse_solver = np.linalg.solve
            else:
                lse_solver = self.lse_solver

            # prepare some initials
            x = np.asarray(initial_point)
            if self.initial_hessian is None:
                self._smoothed_hessian = np.identity(x.size)
            else:
                self._smoothed_hessian = self.initial_hessian

            self._nfev = 0

            # if blocking is enabled we need to keep track of the function values
            if self.blocking:
                fx = loss(x)

                self._nfev += 1
                if self.allowed_increase is None:
                    self.allowed_increase = 2 * self.estimate_stddev(loss, x)

            logger.info("=" * 30)
            logger.info("Starting SPSA optimization")
            if self.blocking:
                self._dump = (eta, eps, lse_solver, x, fx)
            else:
                self._dump = (eta, eps, lse_solver, x, None)
            self.num_step += 1

        else:
            eta, eps, lse_solver, x, fx = self._dump
            iteration_start = time()
            # compute update
            update = self._compute_update(loss, x, self.num_step, next(eps), lse_solver)

            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update

            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = loss(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
                            False,
                        )  # not accepted

                    logger.info(
                        "Iteration %s/%s rejected in %s.",
                        self.num_step,
                        self.maxiter + 1,
                        time() - iteration_start,
                    )
                    return
                fx = fx_next

            logger.info(
                "Iteration %s/%s done in %s.", self.num_step, self.maxiter + 1, time() - iteration_start
            )

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = loss(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
                    True,
                )  # accepted

            # update parameters
            x = x_next
            self._dump = (eta, eps, lse_solver, x, fx)
