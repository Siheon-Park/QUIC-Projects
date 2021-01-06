from sys import flags
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class Sklearn_DataLoader(object):
    """
        Dataset loader for iris, wine data provided in Scikit-learn package

        Attrbutes:
            lb - LabelBinarizer()
            yhot - one-hot encoded label of original data
            data - transformed data
            label - transformed binary label (0, 1)
    """
    def __init__(self, dataset:str='iris', mean:np.ndarray=0, normalize:int=2, labels:tuple=None) -> None:
        """
            Args:
                dataset:str='iris' - type of dataset. i.e. 'iris', 'wine'
                mean:np.ndarray=0 - set the avarge point of the data. default is origin
                normalize:int=2 - normalize each data. see argument ord of numpy.linalg.norm.
                labels:tuple=None - labels to be considered. if None(default), select all labels in data
        """
        super().__init__()
        if dataset=='iris':
            ds = datasets.load_iris
        elif dataset=='wine':
            ds = datasets.load_wine
        else:
            raise NotValidDataTypeError
        data, label = ds(return_X_y=True)

        if labels is not None:
            _temp = [t in labels for t in label]
            data = data[_temp]
            label = label[_temp]
        
        self.limit_num_data = len(label)

        self.lb = LabelBinarizer()
        self.yhot = self.lb.fit_transform(label)
        data = data-(np.mean(data, axis=0)+mean)
        if normalize is not None:
            data = (data.T/np.linalg.norm(data, axis=1, ord=normalize)).T
        self.data = data

    def __call__(self, num_data:int, true_hot:int, shuffle:bool=True):
        """
            Args:
                num_data:int - number of training data
                true_hot:int - label to be encoded as 1
                shuffle:bool=True - to shuffle training, test dataset
            
            return:
                X, y, Xt, yt - training data, training label, test data, test label
        """
        assert num_data>0
        self.label = self.yhot[:,true_hot].flatten()
        if num_data < self.limit_num_data:
            X, Xt, y, yt = train_test_split(self.data, self.label, train_size = num_data, shuffle=shuffle, stratify=self.yhot)
        else:
            X = self.data
            y = self.label
            Xt = None
            yt = None
        return X, y, Xt, yt

class NotValidDataTypeError:
    pass