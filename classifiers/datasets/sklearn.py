from typing import List, Union
import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from seaborn import pairplot

class SklearnDataset(object):
    def __init__(self, loader:Union[callable, str], feature_range: List[float] = None, true_hot: Union[int, str] = None):
        super().__init__()
        if isinstance(loader, str):
            if loader=='iris':
                loader=datasets.load_iris
            elif loader=='wine':
                loader=datasets.load_wine
        [self.__setattr__(name, value) for name, value in loader().items()]
        self.num_features = len(self.feature_names)
        self.num_total = len(self.target)
        if feature_range is not None:
            if len(feature_range) != 2:
                raise ValueError("feature_range should be [lower, upper]")
            self.feature_range = feature_range
            scaler = MinMaxScaler(feature_range=feature_range)
            self.data = scaler.fit_transform(self.data)
            for i in range(self.num_features):
                self.feature_names[i] += f" -> [{np.round(feature_range[0], 2)}, {np.round(feature_range[1], 2)})"
        if true_hot is not None:
            lb = LabelBinarizer()
            if isinstance(true_hot, str):
                try:
                    true_hot = list(self.target_names).index(true_hot)
                except ValueError as e:
                    raise ValueError(str(e)[:-4] + 'target_names')
            elif isinstance(true_hot, int):
                pass
            else:
                raise ValueError("true_hot shoud be 'str' or 'int'")
            self.true_hot = true_hot
            list_target_name = list(self.target_names)
            true_hot_name = list_target_name.pop(true_hot)
            self.target_names = np.array([', '.join(list_target_name), true_hot_name])
            self.target = lb.fit_transform(self.target)[:, true_hot]
        self.frame = DataFrame(data=dict(**dict(zip(self.feature_names, self.data.T)),
                                         target_name=list(map(self.target_names.__getitem__, self.target)),
                                         target=self.target,
                                         ))
    def sample(self, n: int, *args, return_X_y: bool = False, **kwargs):
        if not return_X_y:
            return self.frame.sample(n, *args, **kwargs)
        else:
            _tmp = self.frame.sample(n)
            X = _tmp[_tmp.columns[:self.num_features]].to_numpy()
            y = _tmp[_tmp.columns[-1]].to_numpy()
            return X, y

    def sample_training_and_test_dataset(self, size: tuple[int, int], *args, return_X_y: bool = False, **kwargs):
        _tmp = self.frame.sample(size[0]+size[1], *args, **kwargs)
        training, test = _tmp.iloc[:size[0]], _tmp.iloc[size[0]:]
        if return_X_y:
            X = training[training.columns[:self.num_features]].to_numpy()
            y = training[training.columns[-1]].to_numpy()
            Xt = test[test.columns[:self.num_features]].to_numpy()
            yt = test[test.columns[-1]].to_numpy()
            return X, y, Xt, yt
        else:
            return training, test

    def plot(self, sampled_frame: Union[DataFrame, List[np.ndarray]] = None):
        if sampled_frame is None:
            sampled_frame = self.frame
        elif isinstance(sampled_frame, DataFrame):
            pass
        else:
            X, y = sampled_frame
            sampled_frame = DataFrame(data=dict(**dict(zip(self.feature_names, X.T)),
                                                target_name=list(map(self.target_names.__getitem__, y)),
                                                target=y,
                                                ))
        return pairplot(sampled_frame, hue="target_name", hue_order=self.target_names, diag_kind='hist',
                        x_vars=self.feature_names, y_vars=self.feature_names)