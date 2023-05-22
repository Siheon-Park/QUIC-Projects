import numpy as np
import cvxopt
from sklearn.metrics import accuracy_score
from itertools import product, starmap
import logging
from ._cvxopt_helpers_ import _Matrix_Helper
from . import ConvexClassifier, ConvexError
from .. import process_info_from_alpha
from ..kernel import Kernel
from typing import List, Tuple, Union, Dict, Any

_EPS = 1e-5

logger = logging.getLogger(__name__)


# noinspection PyMissingConstructor
class BinarySVM(ConvexClassifier):
    def __init__(self, kernel: Kernel, C: float = None, mutation: str = 'SVM', **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        _options = ['SVM', 'QASVM', 'REDUCED_SVM', 'REDUCED_QASVM', 'REDUCED_primal_SVM', 'REDUCED_primal_QASVM',
                    'uniform_QASVM', 'REDUCED_uniform_QASVM']
        if mutation not in _options:
            raise ConvexError('Expect one of {:}, received {:}'.format(_options, mutation))
        if 'REDUCED' in mutation:
            self.k = kwargs.get('k', 1)
        self.mutation = mutation
        self.status = None
        self.iterations = None
        self.alpha = None
        self.result = None

    def f(self, test: np.ndarray):
        if len(test.shape) == 1:
            return self.b + sum(self.alpha * self.polary * np.array([self.kernel(test, x) for x in self.data]))
        else:
            return np.array([self.f(xt) for xt in test])

    def predict(self, test: np.ndarray):
        return self.f(test) > 0

    def accuracy(self, test: np.ndarray, testlabel: np.ndarray):
        return accuracy_score(self.predict(test), testlabel)

    # noinspection PyAttributeOutsideInit
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.debug(repr(self))
        super().__init__(X, y)
        self.polary = 2 * y - 1
        if 'uniform' in self.mutation:
            self._fit_uniform()
        else:
            self._fit_cvxopt()
        self._fit_postprocessing()
        logger.debug(self.result)

    def __repr__(self) -> str:
        str_list = [str(self), f'Kernel: {self.kernel}', f'HyperParameter: {self.C}',
                    f'Optimization Status: {self.status}', f'Iterations: {self.iterations}', f'alpha: {self.alpha}',
                    '\n']
        return '\n\t'.join(str_list)

    def __str__(self) -> str:
        return f'BinarySVM: ({self.mutation})'

    def _fit_cvxopt(self):
        logger.info('CVXOPT fitting starting')
        K, Y = _Matrix_Helper.__find_kernel_matrix__(self.data, self.polary, self.kernel)
        if 'REDUCED' in self.mutation:
            P = (K + 1 / self.k) * Y
        else:
            P = K * Y
        P = cvxopt.matrix(P, (self.num_data, self.num_data), 'd')
        self.P = P
        if self.mutation == 'REDUCED_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_SVM__(P, self.num_data, self.C)
        elif self.mutation == 'REDUCED_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_QASVM__(P, self.num_data, self.C)
        elif self.mutation == 'SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__SVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation == 'QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__QASVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation == 'REDUCED_primal_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_PRIMAL_SVM__(P, self.num_data, self.C)
        elif self.mutation == 'REDUCED_primal_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_PRIMAL_QASVM__(P, self.num_data, self.C)
        else:
            P, q, G, h, A, b = (None, None, None, None, None, None)

        cvxopt.solvers.options['show_progress'] = False
        if A is None:
            sol = cvxopt.solvers.qp(P, q, G, h)
        else:
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.status = sol['status']
        self.iterations = sol['iterations']
        self.alpha = np.array(sol['x']).flatten()
        if 'primal' in self.mutation:
            self.alpha = self.alpha[:self.num_data]
        self.result = sol

    def _fit_uniform(self):
        logger.info('CVXOPT fitting skipped for being uniform svm')
        self.status = 'uniform'
        self.iterations = 0
        self.alpha = np.ones(self.num_data) / self.num_data
        self.result = {'x': self.alpha, 'status': self.status, 'iterations': self.iterations}

    def _fit_postprocessing(self):
        if 'REDUCED' not in self.mutation:
            if self.C is not None:
                _temp = np.argwhere(self.alpha > self.C * _EPS).flatten()
            else:
                _temp = np.argwhere(self.alpha > _EPS).flatten()
            b = 0
            for ind in _temp:
                b += self.polary[ind] - sum(
                    self.alpha * self.polary * np.array([self.kernel(self.data[ind], x) for x in self.data]))
            if b != 0:
                self.b = b / len(_temp)
            else:
                self.b = b
        else:
            self.b = np.sum(self.alpha * self.polary) / self.k
        logger.debug('bias b for {:}: {:.10f}'.format(self.mutation, self.b))

        self.support_, self.support_vectors_, self.n_support_ = process_info_from_alpha(self.alpha, self.data)

    @property
    def dual_objective_value(self):
        if self.mutation == 'REDUCED_QASVM':
            K = np.array(list(starmap(self.kernel, product(self.data, repeat=2)))).reshape(self.num_data,
                                                                                           self.num_data) + 1 / self.k
            A = (self.alpha * self.polary).reshape(-1, 1)
            return float(A.T @ K @ A)
        elif self.mutation == 'QASVM':
            K = np.array(list(starmap(self.kernel, product(self.data, repeat=2)))).reshape(self.num_data, self.num_data)
            A = (self.alpha * self.polary).reshape(-1, 1)
            return float(A.T @ K @ A)
        elif self.mutation == 'REDUCED_SVM':
            K = np.array(list(starmap(self.kernel, product(self.data, repeat=2)))).reshape(self.num_data,
                                                                                           self.num_data) + 1 / self.k
            A = (self.alpha * self.polary).reshape(-1, 1)
            return 0.5 * float(A.T @ K @ A) - np.sum(self.alpha)
        elif self.mutation == 'SVM':
            K = np.array(list(starmap(self.kernel, product(self.data, repeat=2)))).reshape(self.num_data, self.num_data)
            A = (self.alpha * self.polary).reshape(-1, 1)
            return 0.5 * float(A.T @ K @ A) - np.sum(self.alpha)
        else:
            return None

class CvxSoftQASVM(ConvexClassifier):
    def __init__(self, kernel: Union[Kernel, str], C: float = None, lamda: float = 1, **kwargs) -> None:
        """ Soft QASVM with cvxopt

        Args:
            kernel (Union[Kernel, str]): kernel function. if it is 'precomputed', the kernel matrix should be given as data in fit method.
            C (float, optional): Hyperparmeter. Defaults to None = inf.
            lamda (float, optional): Hyperparmeter. Defaults to 1.
        """
        self.kernel = kernel
        self.C = C
        self.lamda = lamda

        self.status = None
        self.iterations = None
        self.alpha = None
        self.result = None

    def f(self, test: np.ndarray):
        """ Decision function

        Args:
            test (np.ndarray): test data. If kernel is 'precomputed', test should be kernel matrix of test data and training data.

        Returns:
            fval: Decision function value len(fval) == number of test data
        """
        if self.kernel == 'precomputed':
            return self.b + (self.alpha*self.polary) @ test.T
        else:
            if len(test.shape) == 1:
                return self.b + sum(self.alpha * self.polary * np.array([self.kernel(test, x) for x in self.data]))
            else:
                return np.array([self.f(xt) for xt in test])

    def predict(self, test: np.ndarray):
        """ estimate label of test data

        Args:
            test (np.ndarray): test data. If kernel is 'precomputed', test should be kernel matrix of test data and training data.

        Returns:
            estimated_label: estimated label of test data
        """
        return np.where(self.f(test)>0, 1, 0)

    def accuracy(self, test: np.ndarray, testlabel: np.ndarray):
        """ accuracy score """
        return accuracy_score(self.predict(test), testlabel)

    def score(self, test: np.ndarray, testlabel: np.ndarray):
        """ accuracy score """
        return self.accuracy(test, testlabel)

    # noinspection PyAttributeOutsideInit
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ fit the model

        Args:
            X (np.ndarray): Training data. If kernel is 'precomputed', X should be the kernel matrix.
            y (np.ndarray): binary label (0 or 1)
        """
        logger.debug(repr(self))
        # super().__init__(X, y)
        self.num_data = len(y)
        self.data = X
        self.label = y
        self.polary = 2 * y - 1
        self._fit_cvxopt()
        self._fit_postprocessing()
        logger.debug(self.result)

    def _fit_cvxopt(self):
        logger.info('CVXOPT fitting starting')
        K, Y = _Matrix_Helper.__find_kernel_matrix__(self.data, self.polary, self.kernel)
        if self.C is None:
            P = (K + 1 / self.lamda) * Y
        else:
            P = (K + 1 / self.lamda) * Y + (1/self.C)*np.eye(self.num_data)
        P = cvxopt.matrix(P, (self.num_data, self.num_data), 'd')
        self.P = P
        P, q, G, h, A, b = _Matrix_Helper.__find_matrix__SoftQASVM__(P, self.num_data, self.C)

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(2*P, q, G, h, A, b)

        self.status = sol['status']
        self.iterations = sol['iterations']
        self.alpha = np.array(sol['x']).flatten()
        self.result = sol

    def _fit_postprocessing(self):
        self.b = np.sum(self.alpha * self.polary) / self.lamda
        self.support_, self.support_vectors_, self.n_support_ = process_info_from_alpha(self.alpha, self.data)

    @property
    def dual_objective_value(self):
        Alpha = self.alpha.reshape(-1, 1)
        return float(Alpha.T @ np.array(self.P) @ Alpha)
    
    @property
    def objective_value(self):
        """Optimized objective value

        Returns:
            float: Optimized objective value
        """
        return self.dual_objective_value
