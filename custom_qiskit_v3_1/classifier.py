import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import pickle

import cvxopt

_EPS = 1e-5

class Kernel:
    def __init__(self, kind:str='Pow2', gamma:float=None) -> None:
        self.kind = kind
        self.gamma = gamma

    def __call__(self, X, Y):
        if self.kind == 'RBF':
            kernel = lambda X, Y: np.exp(-self.gamma/2*np.linalg.norm(X-Y)**2)

        elif self.kind == 'Pow2':
            kernel = lambda X, Y: np.abs(X @ Y.T)**2

        elif self.kind == 'linear':
            kernel = lambda X, Y: X @ Y.T

        else:
            kernel = None

        return kernel(X, Y)

    def __repr__(self) -> str:
        return self.kind
        
class Classifier:
    def __init__(self, data:np.ndarray, label:np.ndarray):
        self.data = data
        self.label = label
        self.num_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.alpha = None
        self.name = type(self).__name__
        if self.num_data != label.size:
            raise InvalidDataError('Not enough/More number of labels compare to dataset')

    def plot(self, option:str='sv', axes=plt, **kwargs):
        a = kwargs.get('a', (0, 1))
        cmap = kwargs.get('cmap', plt.cm.coolwarm)# pylint: disable=no-member
        s = kwargs.get('s', 100)
        linewidth = kwargs.get('linewidth', 1.0)
        facecolors = kwargs.get('facecolor', 'none')
        edgecolors = kwargs.get('edgecolors', 'k')

        if 'tsne' in option:
            data = self.data_emb
            a = (0, 1)
        else:
            data = self.data

        if 'data' in option:
            axes.scatter(data[:,a[0]], data[:,a[1]], c=self.label, cmap=cmap)
        elif 'sv' in option:
            support_vector = data[self.support_]
            axes.scatter(data[:,a[0]], data[:,a[1]], c=self.label, cmap=cmap)
            axes.scatter(support_vector[:,a[0]], support_vector[:,a[1]], s=s, linewidth=linewidth, facecolors=facecolors, edgecolors=edgecolors)
        elif 'density' in option:
            sc = axes.scatter(data[:,a[0]], data[:,a[1]], c=self.alpha*self.label)
            if axes==plt:
                plt.colorbar()
            else:
                plt.colorbar(sc, ax=axes)
        elif 'alpha' in option:
            sc = axes.plot(self.alpha)
        else:
            raise ValueError

        if axes==plt:
            axes.title(f'{self.name}')
        else:
            axes.set_title(f'{self.name}')
        axes.grid()

    def tsne(self, perp:float=30):
        return TSNE(n_components=2, perplexity=perp).fit_transform(self.data)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            cls = pickle.load(f)
        return cls

class BinarySVM(Classifier):
    def __init__(self, kernel:Kernel, C:float=None, mutation:str='SVM')->None:
        self.kernel = kernel
        self.C = C
        _options = ['SVM', 'QASVM', 'REDUCED_SVM', 'REDUCED_QASVM']
        assert mutation in _options
        self.mutation = mutation

    def __repr__(self) -> str:
        str_list=[]
        str_list.append(f'{self.name}:')
        str_list.append(f'\n\tKernel: {self.kernel}')
        str_list.append(f'\n\tHyperParameter: {self.C}')
        str_list.append(f'\n\tOptimization Status: {self.status}')
        str_list.append(f'\n\tMutation: {self.mutation}')
        str_list.append(f'\n\tIterations: {self.iterations}')
        return ''.join(str_list)

    def predict(self, test:np.ndarray):
        prediction = []
        for xt in test:
            _temp = self.b + sum(self.alpha*self.polary*np.array([self.kernel(xt, x) for x in self.data]))
            if _temp >0:
                prediction.append(1.0)
            else:
                prediction.append(0.0)
        return np.array(prediction)

    def accuracy(self, test:np.ndarray, testlabel:np.ndarray):
        return accuracy_score(self.predict(test), testlabel)

    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        Classifier.__init__(self, X, y)
        self.polary = 2*y-1
        if self.mutation=='REDUCED_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_SVM__(self)
        elif self.mutation=='REDUCED_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_QASVM__(self)
        elif self.mutation=='SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__SVM__(self)
        elif self.mutation=='QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__QASVM__(self)
        else:
            P, q, G, h, A, b = None
        
        cvxopt.solvers.options['show_progress'] = True
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.status = sol['status']
        self.iterations = sol['iterations']
        self.alpha = np.array(sol['x']).flatten()

        if self.C is not None:
            self.support_ = np.argwhere(self.alpha>self.C*_EPS).flatten()
        else:
            self.support_ = np.argwhere(self.alpha>_EPS).flatten()
        self.support_vectors_ = self.data[self.support_]
        self.n_support_ = len(self.support_)

        if 'REDUCED' not in self.mutation:
            if self.C is not None:
                _temp = np.argwhere((self.alpha>self.C*_EPS) & (self.alpha<self.C*(1-_EPS))).flatten()
            else:
                _temp = np.argwhere(self.alpha>_EPS).flatten()
            b = 0
            for ind in _temp:
                b += self.polary[ind] - sum(self.alpha*self.polary*np.array([self.kernel(self.data[ind], x) for x in self.data]))
            if b is not 0:
                self.b = b/len(_temp)
            else:
                self.b = b
        else:
            self.b = np.sum(self.alpha*self.polary)

class _Matrix_Helper:

    @staticmethod
    def __find_matrix__REDUCED_SVM__(cls):
        n = cls.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = (1+cls.kernel(cls.data[i], cls.data[j]))*cls.polary[i]*cls.polary[j]
        P = cvxopt.matrix(P, (n, n), 'd')
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        if cls.C is None:
            h = cvxopt.matrix(0.0, (n,1), 'd')
            G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = cls.C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1), 'd')
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n), 'd')
        return P, q, G, h, None, None

    @staticmethod
    def __find_matrix__REDUCED_QASVM__(cls):
        n = cls.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = (1+cls.kernel(cls.data[i], cls.data[j]))*cls.polary[i]*cls.polary[j]
        P = cvxopt.matrix(P, (n, n), 'd')
        A = cvxopt.matrix(1.0, (1, n), 'd')
        h = cvxopt.matrix(0.0, (n, 1), 'd')
        G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        if cls.C is None:
            b = cvxopt.matrix(1.0, (1, 1), 'd')
            q = cvxopt.matrix(0.0, (n, 1), 'd')
        else:
            b = cvxopt.matrix(cls.C, (1, 1), 'd')
            q = cvxopt.matrix(-1.0, (n, 1), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__SVM__(cls):
        n = cls.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = cls.kernel(cls.data[i], cls.data[j])*cls.polary[i]*cls.polary[j]
        P = cvxopt.matrix(P, (n, n), 'd')
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        A = cvxopt.matrix(cls.polary, (1, n), 'd')
        b = cvxopt.matrix(0.0, (1, 1), 'd')
        if cls.C is None:
            h = cvxopt.matrix(0.0, (n,1), 'd')
            G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = cls.C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1), 'd')
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__QASVM__(cls):
        n = cls.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = cls.kernel(cls.data[i], cls.data[j])*cls.polary[i]*cls.polary[j]
        P = cvxopt.matrix(P, (n, n), 'd')
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        A1 = cls.polary
        A2 = np.ones(n) if cls.C is None else cls.C*np.ones(n)
        A = cvxopt.matrix(np.vstack([A1, A2]), (2, n), 'd')
        b = cvxopt.matrix([0.0, 1.0], (2, 1), 'd')
        h = cvxopt.matrix(0.0, (n,1), 'd')
        G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        return P, q, G, h, A, b

# error class
class InvalidDataError:
    pass