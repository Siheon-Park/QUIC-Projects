import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 

from qiskit import QiskitError

_EPS = 1e-8
_MU = 1e-3
kernel = lambda X, Y: np.abs(X @ Y.T)**2
H_cbin = lambda p, q: -p*np.log(q)-(1-p)*np.log(q)
sigmoid = lambda x: 1/(1+np.exp(-x))

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

    def plot(self, axes=plt, **kwargs):
        a = kwargs.get('a', (0, 1))
        cmap = kwargs.get('cmap', plt.cm.coolwarm)# pylint: disable=no-member
        s = kwargs.get('s', 100)
        linewidth = kwargs.get('linewidth', 1.0)
        facecolors = kwargs.get('facecolor', 'none')
        edgecolors = kwargs.get('edgecolors', 'k')
        option = kwargs.get('option', 'sv')

        if option=='sv':
            support_vector = self.data[self.alpha>=np.mean(self.alpha)]
            axes.scatter(self.data[:,a[0]], self.data[:,a[1]], c=self.label, cmap=cmap)
            axes.scatter(support_vector[:,a[0]], support_vector[:,a[1]], s=s, linewidth=linewidth, facecolors=facecolors, edgecolors=edgecolors)
        elif option=='density':
            sc = axes.scatter(self.data[:,a[0]], self.data[:,a[1]], c=self.alpha*self.label)
            if axes==plt:
                plt.colorbar()
            else:
                plt.colorbar(sc, ax=axes)
        elif option=='alpha':
            sc = axes.plot(self.alpha)
        else:
            raise ValueError

        if axes==plt:
            axes.title(f'{self.name}')
        else:
            axes.set_title(f'{self.name}')
        axes.grid()


class SVM(Classifier):
    ''' SVM machine to calculate ground truth duel coefficients '''
    def __init__(self, data:np.ndarray, label:np.ndarray, C:int=None):
        super().__init__(data, label)
        self.C = C

    def optimize(self, initial_point:np.ndarray, method:str='SLSQP', **options):
        ''' optimize with scipy
            min_alpha (alpha.T Q alpha - ||alpha||1)
            0<= alpha <= C
            alpha.T y = 0
            '''
        bnds = tuple([(0, self.C) for i in range(self.num_data)])
        cnts = {'type':'eq', 'fun':self.IZZval}
        ret = sp.optimize.minimize(self.objective_function, initial_point, method=method, bounds = bnds, constraints=cnts, options=options)
        self.opt_result = ret
        self.alpha = ret.x

    def qiskit_optimize(self, initial_point:np.ndarray, optimizer, **options):
        """ ref: https://qiskit.org/textbook/ch-applications/vqe-molecules.html"""
        ret = optimizer.optimize(self.num_data, objective_function=self.qiskit_objective_function, initial_point=initial_point, **options)
        self.opt_result = ret
        self.alpha = ret[0]

    def objective_function(self, alpha:np.ndarray):
        ''' quadratic programming 
            f = sum(alpa_i alpha_j y_i y_j k(X_i, X_j) ) - sum(alpha)
        '''
        ret = 0.5*self.ZZZval(alpha)-np.sum(alpha)
        return ret

    def qiskit_objective_function(self, alpha:np.ndarray):
        ''' quadratic programming 
            f = sum(alpa_i alpha_j y_i y_j k(X_i, X_j) ) - sum(alpha)
        '''
        constrain = np.log(abs(self.IZZval(alpha))+_EPS)/_MU - sum(np.log(alpha+_EPS))*_MU - sum(np.log(self.C-alpha+_EPS))*_MU  # for the sake of constraint
        return self.objective_function(alpha) + constrain

    def ZZZval(self, alpha:np.ndarray):
        ''' alpha.T Q alpha, Q_ij = y_i y_j k(X_i, X_j) '''
        ZZZ = (alpha*self.label).reshape(1, -1) @ kernel(self.data, self.data) @ (alpha*self.label).reshape(-1, 1)
        return ZZZ[0, 0]

    def IZZval(self, alpha:np.ndarray):
        ''' |alpha.T y|^2 '''
        ZZ = np.abs(np.sum(self.label*alpha))**2
        return ZZ

    def ZZval(self, alpha:np.ndarray, test:np.ndarray):
        ''' alpha y kernel(x, x') '''
        ZZ = (alpha*self.label).reshape(1, -1) @ kernel(self.data, test)
        return ZZ.reshape(-1)

    def IZval(self, alpha:np.ndarray):
        ''' alpha.T y '''
        Z = np.sum(self.label*alpha)
        return Z

    def predict(self, test:np.ndarray, alpha:np.ndarray=None):
        if not isinstance(alpha, np.ndarray):
            alpha = self.alpha
        ZZ = (alpha*self.label).reshape(1, -1) @ kernel(self.data, test)
        est_y = np.sign( ZZ.reshape(-1) )
        return est_y

    def check_performance(self, test:np.ndarray, testlabel:np.ndarray, alpha:np.ndarray=None):
        if not isinstance(alpha, np.ndarray):
            alpha = self.alpha
        est_y = self.predict(test, alpha)
        return sum((est_y==testlabel).reshape(-1))/len(testlabel)

# error class
class InvalidDataError(QiskitError):
    pass