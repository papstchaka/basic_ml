import numpy as np
import abc

class _classifier(abc.ABC):
    
    @abc.abstractmethod
    def train(self) -> None:
        pass
    
    @abc.abstractmethod
    def predict(self) -> None:
        pass
        
    def calc_rates(self, y_test:np.array, y_pred:np.array) -> list:
        '''
        calculates the true positives, the false positives, the true negatives and the false negatives each class
        Parameters:
            - y_test: actual y-values - ground truth [numpy.array]
            - y_pred: predicted y-values - prediction [numpy.array]
        Returns:
            - rates: List, each element containing the true positive, the false positive, the true negative and the false negative rate of respective class [List]
        '''
        rates = []
        sum_samples = y_test.__len__()
        classes = np.unique(y_test)
        for _class in classes:
            _pt = np.where(y_test == _class)
            _nt = np.where(y_test != _class)
            _pp = np.where(y_pred == _class)
            _np = np.where(y_pred != _class)

            tp = np.sum([_pt_ in _pp[0] for _pt_ in _pt[0]])
            fp = np.sum([_pt_ not in _pp[0] for _pt_ in _pt[0]])
            tn = np.sum([_nt_ in _np[0] for _nt_ in _nt[0]])
            fn = np.sum([_nt_ not in _np[0] for _nt_ in _nt[0]])
            rates.append((tp, fp, tn, fn))
        return rates
    
    @abc.abstractmethod
    def score(self, y_test:np.array, y_pred:np.array, mode:str = "accuracy", average:bool = True) -> float:
        '''
        implements the scoring function for the classifiers, depending on given mode
        Parameters:
            - y_test: actual y-values - ground truth [numpy.array]
            - y_pred: predicted y-values - prediction [numpy.array]
            - mode: mode of the scoring function. Possible values are [String]
                - Recall --> "recall"
                - Precision --> "precision"
                - Accuracy --> "accuracy" = default
                - F1 --> "f1"
                - balanced Accuracy --> "balanced_accuracy"
            - average: whether (=True, default) or not (=False) to average the estimated score (=metric) over the given classes
        Returns:
            - score: calculated score [Float]
        '''
        ## make sure to have as many predicted data points as actual ones
        assert(y_test.__len__() == y_pred.__len__())
        ## make sure y_test and y_pred are actually numpy.arrays
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        rates = self.calc_rates(y_test, y_pred)
        if mode == "recall":
            metric = [tp / (tp + fn) for (tp, fp, tn, fn) in rates]
        elif mode == "precision":
            metric = [tp / (tp + fp) for (tp, fp, tn, fn) in rates]
        elif mode == "accuracy":
            metric = [(tp + tn) / (tp + tn + fp + fn) for (tp, fp, tn, fn) in rates]
        elif mode == "f1":
            metric = [2*tp / (2*tp + fp + fn) for (tp, fp, tn, fn) in rates]
        elif mode == "balanced_accuracy":
            metric = [(tp / (2*(tp + fn))) + (tn / (2*(tn + fp))) for (tp, fp, tn, fn) in rates]
        else:
            print('Unknown score function. Accuracy is used')    
            metric = [(tp + tn) / (tp + tn + fp + fn) for (tp, fp, tn, fn) in rates]
        if average:
            metric = np.sum(metric) / metric.__len__()
        return metric
    
class _regressor(abc.ABC):
    
    @abc.abstractmethod
    def train(self) -> None:
        pass
    
    @abc.abstractmethod
    def score(self, y_test:np.array, y_pred:np.array, mode:str = "l2") -> float:
        '''
        implements the scoring function for the regressors, depending on given mode
        Parameters:
            - y_test: actual y-values - ground truth [numpy.array]
            - y_pred: predicted y-values - prediction [numpy.array]
            - mode: mode of the scoring function. Possible values are [String]
                - L1-norm Loss --> "l1"
                - L2-norm Loss --> "l2" = default
                - Mean squared Error --> "mse"
                - Mean absolute Error --> "mae"
                - Root mean squared Error --> "rmse"
        Returns:
            - score: calculated score [Float]
        '''
        ## make sure to have as many predicted data points as actual ones
        assert(y_test.__len__() == y_pred.__len__())
        ## make sure y_test and y_pred are actually numpy.arrays
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        if mode == "l1":
            return np.sum( np.abs(y_test - y_pred) )
        elif mode == "l2":
            return np.sum(( y_test - y_pred )**2 )
        elif mode == "mse":
            return np.sum(( y_test - y_pred )**2 ) / y_test.__len__()
        elif mode == "mae":
            return np.sum( np.abs(y_test - y_pred) ) / y_test.__len__()
        elif mode == "rmse":
            return np.sqrt(np.sum(( y_test - y_pred )**2 ) / y_test.__len__())
        else:
            print('Unknown score function. L2 is used')
            return np.sum(( y_test - y_pred )**2 )