import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 

from sklearn.manifold import TSNE
from qiskit import QiskitError

import cvxopt

_EPS = 1e-5
_MU = 1e-3
kernel = lambda X, Y: np.abs(X @ Y.T)**2
H_cbin = lambda p, q: -p*np.log(q)-(1-p)*np.log(1-q)
sigmoid = lambda x: 1/(1+np.exp(-x))

def Pow2_kernel(X, Y, gamma):
    return np.dot(X, Y)**2

def linear_kernel(X, Y, gamma):
    return np.dot(X, Y)

def RBF_kernel(X1, X2, gamma):
    return np.exp(-gamma/2*(np.linalg.norm(X1-X2)**2))

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


class SVM(Classifier):
    ''' SVM machine to calculate ground truth duel coefficients '''
    def __init__(self, data:np.ndarray, label:np.ndarray, C:float=None):
        super().__init__(data, label)
        self.C = C if C is None else float(C)

    def optimize(self, kernel, kernel_hyper, option:str):
        if option == 'svm':
            self.svm_optimize(kernel=kernel, kernel_hyper=kernel_hyper)
        elif option == 'qsvm':
            self.qsvm_optimize(kernel=kernel, kernel_hyper=kernel_hyper)
        else:
            pass

    def svm_optimize(self, kernel, kernel_hyper):
        self.kernel = kernel
        self.kernel_hyper = kernel_hyper
        n = self.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = 1+self.kernel(self.data[i], self.data[j], self.kernel_hyper)*self.label[i]*self.label[j]
        P = cvxopt.matrix(P, (n, n))
        q = cvxopt.matrix(-1.0, (n, 1))
        if self.C is None:
            h = cvxopt.matrix(np.zeros(n))
            G = cvxopt.matrix(-np.eye(n))
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = self.C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1))
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n))
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, None, None)

        self.alpha = np.array(sol['x']).flatten()
        if self.C is not None:
            self.support_ = np.argwhere(self.alpha>self.C*_EPS).flatten()
        else:
            self.support_ = np.argwhere(self.alpha>_EPS).flatten()
        self.support_vectors_ = self.data[self.support_]
        self.n_support_ = len(self.support_)
        self.b = np.sum(self.alpha*self.label)

    def qsvm_optimize(self, kernel, kernel_hyper):
        assert self.C is not None
        self.kernel = kernel
        self.kernel_hyper = kernel_hyper
        n = self.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = 1+self.kernel(self.data[i], self.data[j], self.kernel_hyper)*self.label[i]*self.label[j]
        P = cvxopt.matrix(P, (n, n))
        q = cvxopt.matrix(-1.0, (n, 1))
        h = cvxopt.matrix(np.zeros(n))
        G = cvxopt.matrix(-np.eye(n))
        A = cvxopt.matrix(np.ones_like(self.label), (1, n))
        b = cvxopt.matrix(self.C, (1, 1))
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.alpha = np.array(sol['x']).flatten()
        self.support_ = np.argwhere(self.alpha>self.C*_EPS).flatten()
        self.support_vectors_ = self.data[self.support_]
        self.n_support_ = len(self.support_)
        self.b = np.sum(self.alpha*self.label)

    def predict(self, test:np.ndarray):
        prediction = []
        for xt in test:
            _temp = self.b + sum(self.alpha*self.label*np.array([self.kernel(xt, x, self.kernel_hyper) for x in self.data]))
            if _temp >0:
                prediction.append(1.0)
            else:
                prediction.append(-1.0)
        return np.array(prediction)

    def accuracy(self, test:np.ndarray, testlabel:np.ndarray):
        return sum(self.predict(test)==testlabel)/len(testlabel)

# error class
class InvalidDataError(QiskitError):
    pass