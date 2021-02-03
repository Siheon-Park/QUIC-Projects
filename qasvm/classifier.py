import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from ._cvxopt_helpers_ import _Matrix_Helper
from .kernel import Kernel
import pickle

import cvxopt

_EPS = 1e-5
        
class Classifier:
    def __init__(self, data:np.ndarray, label:np.ndarray):
        self.data = data
        self.label = label
        self.num_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.alpha = None
        self.name = type(self).__name__
        if self.num_data != label.size:
            raise ValueError('Not enough/More number of labels compare to dataset')

    def plot(self, option:str='sv', ax=plt, a = (0,1), *args, **kwargs):
        if 'tsne' in option:
            data = self.data_emb
            a = (0, 1)
        else:
            data = self.data

        if 'data' in option:
            ax.scatter(data[:,a[0]], data[:,a[1]], c=self.label, *args, **kwargs)
        elif 'sv' in option:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.coolwarm
            support_vector = data[self.support_]
            ax.scatter(data[:,a[0]], data[:,a[1]], c=self.label, cmap = plt.cm.coolwarm)
            ax.scatter(support_vector[:,a[0]], support_vector[:,a[1]], s=100, linewidth=1.0, edgecolors='k', facecolors='none')
        elif 'density' in option:
            sc = ax.scatter(data[:,a[0]], data[:,a[1]], c=self.alpha*self.label, *args, **kwargs)
            if ax==plt:
                plt.colorbar()
            else:
                plt.colorbar(sc, ax=ax)
        elif 'alpha' in option:
            ax.plot(self.alpha, 'k', *args, **kwargs)
            ax.plot(self.support_, self.alpha[self.support_], 'xk', *args, **kwargs)
        else:
            raise ValueError('invalid option')

        if ax==plt:
            ax.title(f'{self.name}')
        else:
            ax.set_title(f'{self.name}')
        ax.grid()

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
    def __init__(self, kernel:Kernel, C:float=None, mutation:str='SVM', **kwargs)->None:
        self.kernel = kernel
        self.C = C
        _options = ['SVM', 'QASVM', 'REDUCED_SVM', 'REDUCED_QASVM', 'REDUCED_primal_SVM', 'REDUCED_primal_QASVM', 'uniform_QASVM', 'REDUCED_uniform_QASVM']
        if mutation not in _options:
            raise ValueError('Expect one of {:}, received {:}'.format(_options, mutation))
        if 'REDUCED' in mutation:
            self.k = kwargs.get('k', 1)
        self.mutation = mutation
        self.status=None
        self.iterations=None

    def __repr__(self) -> str:
        str_list=[]
        str_list.append(f'BinarySVM: ({self.mutation})')
        str_list.append(f'\n\tKernel: {self.kernel}')
        str_list.append(f'\n\tHyperParameter: {self.C}')
        str_list.append(f'\n\tOptimization Status: {self.status}')
        str_list.append(f'\n\tIterations: {self.iterations}')
        str_list.append('\n')
        return ''.join(str_list)

    def f(self, test:np.ndarray):
        if len(test.shape)==1:
            return self.b + sum(self.alpha*self.polary*np.array([self.kernel(test, x) for x in self.data]))
        else:
            return np.array([self.f(xt) for xt in test])

    def predict(self, test:np.ndarray):
        return self.f(test)>0
        
    def accuracy(self, test:np.ndarray, testlabel:np.ndarray):
        return accuracy_score(self.predict(test), testlabel)

    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        super().__init__(X, y)
        self.polary = 2*y-1
        if 'uniform' in self.mutation:
            self._fit_uniform()
        else:
            self._fit_cvxopt()
        self._fit_postprocessing()

    def plot_boundary(self, ax=plt, plot_data:bool=True, fig=None):
        assert self.data.shape[1]==2
        xx = np.linspace(min(self.data[:,0]), max(self.data[:,0]), 100)
        yy = np.linspace(min(self.data[:,1]), max(self.data[:,1]), 10)
        XX, YY = np.meshgrid(xx, yy)
        xxx = XX.flatten()
        yyy = YY.flatten()
        zzz = self.f(np.vstack((xxx,yyy)).T)
        ZZ = zzz.reshape(XX.shape)
        c1 = min(np.abs(self.f(self.data[self.polary>0])))
        c2 = -min(np.abs(self.f(self.data[self.polary<0])))
        cdict = {-1:'b', 0:'k', 1:'r', c1:'m', c2:'c'}
        levels = dict(sorted(cdict.items()))
        CS = ax.contour(XX, YY, ZZ, levels=tuple(levels.keys()), colors=tuple(levels.values()))
        PS = ax.contourf(XX, YY, ZZ, levels = 100, cmap='RdBu')
        ax.clabel(CS, inline=1, fontsize=20)
        if fig is None:
            ax.colorbar(PS)
        else:
            fig.colorbar(PS, ax=ax)
        if plot_data:
            self.plot('sv', ax=ax)

    def _fit_cvxopt(self):
        K, Y = _Matrix_Helper.__find_kernel_matrix__(self.data, self.polary, self.kernel)
        if 'REDUCED' in self.mutation:
            P = (K+1/self.k)*Y
        else:
            P = K*Y
        P = cvxopt.matrix(P, (self.num_data, self.num_data), 'd')
        self.P = P
        if self.mutation=='REDUCED_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_SVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation=='REDUCED_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_QASVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation=='SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__SVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation=='QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__QASVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation=='REDUCED_primal_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_PRIMAL_SVM__(P, self.num_data, self.C, self.polary)
        elif self.mutation=='REDUCED_primal_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_PRIMAL_QASVM__(P, self.num_data, self.C, self.polary)
        else:
            P, q, G, h, A, b = None
        
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
            
    def _fit_uniform(self):
        self.status = 'uniform'
        self.iterations = 0
        self.alpha = np.ones(self.num_data)/self.num_data

    def _fit_postprocessing(self):
        if 'REDUCED' not in self.mutation:
            if self.C is not None:
                _temp = np.argwhere(self.alpha>self.C*_EPS).flatten()
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
            self.b = np.sum(self.alpha*self.polary)/self.k

        self.support_ = np.argwhere(self.alpha>0.01*max(self.alpha)).flatten()
        self.support_vectors_ = self.data[self.support_]
        self.n_support_ = len(self.support_)


