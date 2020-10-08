import numpy as np

def data_generation(dim, num, label, **kwargs):
    cov = kwargs.get('cov', np.eye(dim))
    mean = kwargs.get('mean', np.zeros(dim))
    data = np.random.multivariate_normal(mean, cov, size=num)
    labels = np.ones(num)*label
    return data, labels