import numpy as np
import abc
from .metrics import classifier_score, regressor_score

'''
containing abstract classes as support for the other algorithms - for example implementing the score() function for all classifiers and regressors
'''

class _classifier(abc.ABC):
    '''
    abstract class containing all functions that every classifier have in common -> completely implements the score() function as it is the same for all classifiers
    '''
    
    @abc.abstractmethod
    def _imports(self) -> None:
        '''
        force all classifier to have an _imports function to handle all the needed imports independently.
        Parameters:
            - None
        Returns:
            - None
        '''
        pass
    
    @abc.abstractmethod
    def train(self, X_train:np.array) -> np.array:
        '''
        implements the training function. Makes sure that every classifier has one
        Paramters:
            - X_train: x values to train [numpy.array]
        Returns:
            - X_train: processed x values for training [numpy.array]
        '''
        ## make sure X_train is numpy.array
        X_train = np.array(X_train)
        return X_train
    
    @abc.abstractmethod
    def predict(self, X_test:np.array) -> np.array:
        '''
        implements the prediction function. Makes sure that every classifier has one
        Paramters:
            - X_test: x values to predict [numpy.array]
        Returns:
            - X_test: processed x values for prediction [numpy.array]
        '''
        ## make sure X_test is numpy.array
        X_test = np.array(X_test)
        return X_test
    
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
        return classifier_score(y_test, y_pred, mode)
    
class _regressor(abc.ABC):
    '''
    abstract class containing all functions that every regressor have in common -> completely implements the score() function as it is the same for all regressors
    '''
    
    @abc.abstractmethod
    def _imports(self) -> None:
        '''
        force all regressor to have an _imports function to handle all the needed imports independently.
        Parameters:
            - None
        Returns:
            - None
        '''
        pass
    
    @abc.abstractmethod
    def train(self, X_train:np.array) -> np.array:
        '''
        implements the training function. Makes sure that every regressor has one
        Paramters:
            - X_train: x values to train [numpy.array]
        Returns:
            - X_train: processed x values for training [numpy.array]
        '''
        ## make sure X_train is numpy.array
        X_train = np.array(X_train)
        return X_train
    
    @abc.abstractmethod
    def predict(self, X_test:np.array) -> np.array:
        '''
        implements the prediction function. Makes sure that every regressor has one
        Paramters:
            - X_test: x values to predict [numpy.array]
        Returns:
            - X_test: processed x values for prediction [numpy.array]
        '''
        ## make sure X_test is numpy.array
        X_test = np.array(X_test)
        return X_test
    
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
        return regressor_score(y_test, y_pred, mode)