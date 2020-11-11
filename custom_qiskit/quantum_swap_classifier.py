import numpy as np
import scipy as sp

from qiskit.circuit import QuantumCircuit
from qiskit import execute

import quantum_encoder
from swap_classifier import _SWAP_classifier
from classifier import _EPS, _MU

class quantum_SWAP_classifier(_SWAP_classifier):
    """ swap classifier
        aim of this module is to simulate svm on real device
        cannot get exact alpha
    """
    def __init__(self, data:np.ndarray, label:np.ndarray, id:int, layer:int, backend, execute_option:dict={}):
        super().__init__(data, label)
        self._create_double_qc(id, layer)
        self._create_single_qc(id, layer)
        self.execute_option = execute_option
        self.backend=backend
        self.jobs = []

    def optimize(self, backend, initial_point:np.ndarray, method:str='SLSQP', **options):
        ''' optimize with scipy
            min_alpha (alpha.T Q alpha)
            alpha.T y ~ 0
            '''
        self.backend = backend
        #ret = sp.optimize.minimize(self.objective_function, initial_point, method=method, options=options)
        cnts = {'type':'eq', 'fun':self.IZZval}
        ret = sp.optimize.minimize(self.objective_function, initial_point, method=method, constraints=cnts, options=options)
        self.opt_result = ret
        self._theta = ret.x
        self.alpha = self._bind_parameter_return_alpha(ret.x, self.class_weight_qc, self.class_theta)

    def objective_function(self, theta:np.ndarray):
        count_dict = self.execute(theta, self.backend, **self.execute_option)
        # return self.ZZZval(count_dict) + np.log(abs(self.IZZval(count_dict))+_EPS)/_MU
        return 0.5*self._ZZZval(count_dict)-1

    def qiskit_optimize(self, backend, initial_point:np.ndarray, optimizer, **options):
        """ ref: https://qiskit.org/textbook/ch-applications/vqe-molecules.html """
        self.backend = backend
        ret = optimizer.optimize(len(self.theta1), objective_function=self.objective_function, initial_point=initial_point, **options)
        self.opt_result = ret
        self._theta = ret[0]
        self.alpha = self._bind_parameter_return_alpha(ret[0], self.class_weight_qc, self.class_theta)
    
    def IZZval(self, theta:np.ndarray):
        count_dict = self.execute(theta, self.backend, **self.execute_option)
        return self._IZZval(count_dict)

    def _ZZZval(self, count:dict):
        _temp = 0
        _tot = sum(count.values())
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    _temp+=count.get(str(i)+str(j)+str(k), 0)*(-1)**(i+j+k)
        return _temp/_tot

    def _IZZval(self, count:dict):
        _temp = 0
        _tot = sum(count.values())
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    _temp+=count.get(str(i)+str(j)+str(k), 0)*(-1)**(i+k)
        return _temp/_tot

    def _ZZval(self, count:dict):
        _temp = 0
        _tot = sum(count.values())
        for i in (0, 1):
            for j in (0, 1):
                _temp+=count.get('0'+str(i)+str(j), 0)*(-1)**(i+j)
        return _temp/_tot

    def predict(self, test:np.ndarray):
        qr = self.class_qreg[-1]
        qc = QuantumCircuit(qr)
        qc.encode(test, qr, name=f"testdata")
        classifier_qc = qc.combine(self.class_qc)
        if self.class_theta is not None:
            bind_qc = classifier_qc.bind_parameters({self.class_theta:np.pi*self._theta})
        else:
            bind_qc = classifier_qc
        job = execute(bind_qc, self.backend, **self.execute_option)
        result = job.result()
        return np.sign(self._ZZval(result.get_counts()))


    def check_performance(self, test:np.ndarray, testlabel:np.ndarray):
        est_y = np.array([self.predict(test[i]) for i in range(len(testlabel))])
        return sum((est_y==testlabel).reshape(-1))/len(testlabel)

    def execute(self, params, backend, **kwargs):
        bind_qc = self.qc.bind_parameters({self.theta1:np.pi*params, self.theta2:np.pi*params})
        job = execute(bind_qc, backend, **kwargs)
        self.jobs.append(job)
        result = job.result()
        return result.get_counts()

class quantum_uniform_SWAP_classifier(quantum_SWAP_classifier):
    def __init__(self, data:np.ndarray, label:np.ndarray, backend, execute_option:dict={}):
        super().__init__(data, label, 0, 0, backend, execute_option)
        self.alpha = self._bind_parameter_return_alpha(None, self.class_weight_qc, self.class_theta)

    def optimize(self, backend, initial_point:np.ndarray, method:str='SLSQP', **options):
        pass

    def qiskit_optimize(self, backend, initial_point:np.ndarray, optimizer, **options):
        pass
