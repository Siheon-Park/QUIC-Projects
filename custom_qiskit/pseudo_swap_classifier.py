import numpy as np
import scipy as sp

from swap_classifier import _SWAP_classifier
from classifier import _EPS, _MU
from classifier import H_cbin, sigmoid

class pseudo_SWAP_classifier(_SWAP_classifier):
    """ mimic swap classifier
        aim of this module is to check different constraints on swap classifier compared to classical SVM has critical effect,
        and to simulate quantum circuit in short time.
        need to calculate exact alpha"""
    def __init__(self, data:np.ndarray, label:np.ndarray, id:int=1, layer:int=4):
        super().__init__(data, label)
        self.weight_qc = self.create_entangling_circuit(id, layer)
        self.theta = self.weight_qc.theta

    def optimize(self, initial_point:np.ndarray, method:str='SLSQP', **options):
        ''' optimize with scipy
            min_alpha (alpha.T Q alpha - ||alpha||1)
            alpha.T y = 0
            '''
        cnts = {'type':'eq', 'fun':self.IZZval}
        ret = sp.optimize.minimize(self.objective_function, initial_point, method=method, constraints=cnts, options=options)
        self.opt_result = ret
        self.alpha = self._bind_parameter_return_alpha(ret.x, self.weight_qc, self.theta)

    def qiskit_optimize(self, initial_point:np.ndarray, optimizer, **options):
        """ ref: https://qiskit.org/textbook/ch-applications/vqe-molecules.html
            To check if equality constraint approximation is valid
        """
        ret = optimizer.optimize(len(self.theta), objective_function=self.qiskit_objective_function, initial_point=initial_point, **options)
        self.opt_result = ret
        self.alpha = self._bind_parameter_return_alpha(ret[0], self.weight_qc, self.theta)

    def objective_function(self, theta:np.ndarray):
        ret = 0.5*self.ZZZval(theta)-1 # sum alpha is 1
        return ret

    def qiskit_objective_function(self, theta:np.ndarray):
        ''' quadratic programming 
            f = sum(alpa_i alpha_j y_i y_j k(X_i, X_j) )
        '''
        alpha = self._bind_parameter_return_alpha(theta, self.weight_qc, self.theta)
        constrain = np.log(abs(super().IZZval(alpha))+_EPS)/_MU # for the sake of constraint
        return super().ZZZval(alpha) + constrain

    def ZZZval(self, theta:np.ndarray):
        alpha = self._bind_parameter_return_alpha(theta, self.weight_qc, self.theta)
        return super().ZZZval(alpha)

    def IZZval(self, theta:np.ndarray):
        alpha = self._bind_parameter_return_alpha(theta, self.weight_qc, self.theta)
        return super().IZZval(alpha)

    def ZZval(self, theta:np.ndarray, test:np.ndarray):
        alpha = self._bind_parameter_return_alpha(theta, self.weight_qc, self.theta)
        return super().ZZval(alpha, test)

    def IZval(self, theta:np.ndarray):
        alpha = self._bind_parameter_return_alpha(theta, self.weight_qc, self.theta)
        return super().IZval(alpha)

    def _create_double_qc(self, id, layer):
        pass

    def _create_single_qc(self, id, layer):
        pass

class pseudo_uniform_SWAP_classifier(pseudo_SWAP_classifier):
    def __init__(self, data:np.ndarray, label:np.ndarray):
        super().__init__(data, label, 0, 0)
        self.alpha = np.ones(2**self.index_qubit_num)/(2**self.index_qubit_num)

    def optimize(self, initial_point:np.ndarray, method:str='SLSQP', **options):
        pass

    def qiskit_optimize(self, initial_point:np.ndarray, optimizer, **options):
        pass

    def objective_function(self, alpha: np.ndarray):
        pass

class pseudo_empirical_SWAP_classifier(pseudo_SWAP_classifier):
    """" objective function now use empirical cross entropy """
    def optimize(self, initial_point:np.ndarray, testdata:np.ndarray, testlabel:np.ndarray, method:str='SLSQP', **options):
        ''' optimize with scipy
            min_alpha (alpha.T Q alpha - ||alpha||1)
            alpha.T y = 0
            '''
        self.validation_data = testdata
        self.validation_label = testlabel
        assert testdata.shape[0] == testlabel.size
        cnts = {'type':'eq', 'fun':self.IZval}
        ret = sp.optimize.minimize(self.objective_function, initial_point, method=method, constraints=cnts, options=options)
        self.opt_result = ret
        self.alpha = self._bind_parameter_return_alpha(ret.x, self.weight_qc, self.theta)

    def qiskit_optimize(self, initial_point:np.ndarray, testdata:np.ndarray, testlabel:np.ndarray, optimizer, **options):
        """ ref: https://qiskit.org/textbook/ch-applications/vqe-molecules.html
            To check if equality constraint approximation is valid
        """
        self.validation_data = testdata
        self.validation_label = testlabel
        assert testdata.shape[0] != testlabel.size       
        ret = optimizer.optimize(len(self.theta), objective_function=self.qiskit_objective_function, initial_point=initial_point, **options)
        self.opt_result = ret
        self.alpha = self._bind_parameter_return_alpha(ret[0], self.weight_qc, self.theta)

    def objective_function(self, theta:np.ndarray):
        # TODO
        q = sigmoid(self.ZZval(theta, self.validation_data)).reshape(-1)
        return np.sum(H_cbin((self.label+1)/2, q))/len(self.label)

    def qiskit_objective_function(self, theta:np.ndarray):
        ''' quadratic programming 
            f = sum(alpa_i alpha_j y_i y_j k(X_i, X_j) )
        '''
        # TODO
        pass