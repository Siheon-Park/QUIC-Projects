import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KernelCenterer
from ._cvxopt_helpers_ import _Matrix_Helper
import pickle

import cvxopt

_EPS = 1e-5

class Kernel:
    def __init__(self, kind:str='Pow2', gamma:float=1) -> None:
        poss = ['Pow2', 'RBF', 'linear', 'Phase', 'Cosine']
        if kind not in poss:
            raise ValueError('Expect one of {:}, received {:}'.format(poss, kind))
        self.kind = kind
        self.gamma = gamma

    def __call__(self, X, Y):
        if self.kind == 'RBF':
            kernel = lambda X, Y: np.exp(-self.gamma/2*np.linalg.norm(X-Y)**2)

        elif self.kind == 'Pow2':
            kernel = lambda X, Y: np.abs(X @ Y.T)**2

        elif self.kind == 'linear':
            kernel = lambda X, Y: X @ Y.T

        elif self.kind == 'Phase':
            def PhaseKernel(X, Y):
                assert len(X)==len(Y)
                N = len(X)
                cos = sum([np.cos(X[i]-Y[i]) for i in range(N)])**2
                sin = sum([np.cos(X[i]-Y[i]) for i in range(N)])**2
                return (cos+sin)/N/N
            kernel = PhaseKernel

        elif self.kind == 'Cosine':
            def Cosinekernel(X, Y):
                assert len(X)==len(Y)
                N = len(X)
                cos = np.prod([np.cos(X[i]-Y[i]) for i in range(N)])
                return cos
            kernel = Cosinekernel
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
        _options = ['SVM', 'QASVM', 'REDUCED_SVM', 'REDUCED_QASVM', 'REDUCED_primal_SVM', 'REDUCED_primal_QASVM']
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
            return np.array([self.b + sum(self.alpha*self.polary*np.array([self.kernel(xt, x) for x in self.data])) for xt in test])

    def predict(self, test:np.ndarray):
        return self.f(test)>0
        
    def accuracy(self, test:np.ndarray, testlabel:np.ndarray):
        return accuracy_score(self.predict(test), testlabel)

    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        super().__init__(X, y)
        self.polary = 2*y-1
        if 'REDUCED' in self.mutation:
            P = _Matrix_Helper.__find_REDUCED_kernel_matrix__(self, self.data, self.polary)
        else:
            P = _Matrix_Helper.__find_kernel_matrix__(self, self.data, self.polary)
        # P = cvxopt.matrix(KernelCenterer().fit_transform(np.array(P)), (self.num_data, self.num_data), 'd') # kernel scaling
        if self.mutation=='REDUCED_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_SVM__(P, self)
        elif self.mutation=='REDUCED_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_QASVM__(P, self)
        elif self.mutation=='SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__SVM__(P, self)
        elif self.mutation=='QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__QASVM__(P, self)
        elif self.mutation=='REDUCED_primal_SVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_PRIMAL_SVM__(P, self)
        elif self.mutation=='REDUCED_primal_QASVM':
            P, q, G, h, A, b = _Matrix_Helper.__find_matrix__REDUCED_PRIMAL_QASVM__(P, self)
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
            self.b = np.sum(self.alpha*self.polary)/self.k

        self.support_ = np.argwhere(self.alpha>0.01*max(self.alpha)).flatten()
        self.support_vectors_ = self.data[self.support_]
        self.n_support_ = len(self.support_)

    def plot_boundary(self, ax=plt, plot_data:bool=True):
        assert self.data.shape[1]==2
        xx = np.linspace(min(self.data[:,0]), max(self.data[:,0]), 100)
        yy = np.linspace(min(self.data[:,1]), max(self.data[:,1]), 10)
        XX, YY = np.meshgrid(xx, yy)
        xxx = XX.flatten()
        yyy = YY.flatten()
        ZZ = self.f(np.vstack((xxx,yyy)).T).reshape(XX.shape)
        CS = ax.contour(XX, YY, ZZ, levels=(-1,0,1), colors=('b','k','r'))
        ax.clabel(CS, inline=1, fontsize=20)
        if plot_data:
            self.plot('sv', ax=ax)


