'''
implements preprocessing for data
'''

import numpy as np

def train_test_split(x:np.array, y:np.array = [], train_size:float = 0.75, random_state:int = 42, shuffle:bool = True) -> tuple:
    '''
    Split arrays or matrices into random train and test subsets
    Parameters:
        - x: x array to split [numpy.array]
        - y: y array to split (like x) [numpy.array, default = []]
        - train_size: size of the train set [Float, default = 0.75]
        - random_state: random state for shuffling [Integer, default = 42]
        - shuffle: whether (=True, default) to shuffle or not (=False) [Boolean]
    Returns:
        - tuple containing [Tuple]
            - X_train: x train array [numpy.array]
            - X_test: x test array [numpy.array]
            - y_train: y train array [numpy.array, default = None]
            - y_test: y test array [numpy.array, default = None]
    '''
    ## set random seed
    np.random.seed(random_state)
    ## get the index where dataset gets splitte
    idx = np.ceil(train_size * x.__len__()).astype(int)
    indices = [i for i in range(x.__len__())]
    if shuffle:
        indices = np.random.permutation(x.__len__())
    train = indices[:idx]
    test = indices[idx:]
    ## when y array given
    if y.__len__() > 0:
        ## check length of both arrays
        assert len(x) == len(y), "x and y data streams must have same length"
        X_train = x[train]
        X_test = x[test]
        y_train = y[train]
        y_test = y[test]
        return (X_train, X_test, y_train, y_test)
    X_train = x[train]
    X_test = x[test]
    return (X_train, X_test)

class MinMaxScaler:
    
    '''
    implements a MinMaxScaler that scales all given data into the range from 0 to 1
    '''
    
    def __init__(self) -> None:
        '''
        constructor of class
        initializes:
            - min: Minimum to None
            - max: Maximum to None
            - is_fitted: whether scaler is fitted or not to False [Boolean]
        Returns:
            - None
        '''
        self.min = None
        self.max = None
        self.is_fitted = False
        self._imports()
        
    def _imports(self) -> None:
        '''
        handles the imports for the class - imports needed modules
        Parameters:
            - None
        Returns:
            - None
        '''
        global np
        import numpy
        np = numpy
        
    def check_is_fitted(self) -> None:
        '''
        check whether model is already fitted
        '''
        if self.is_fitted:
            pass
        else:
            raise Exception("scaler is not fitted yet!")
    
    def get_min_max(self, X:np.array):
        '''
        calculates Minimum und Maximum of X values
        Parameters:
            - X: x value data [numpy.array]
        Returns:
            - None
        '''
        self.min = X.min(axis = 0)
        self.max = X.max(axis = 0)
    
    def fit(self, X:np.array) -> None:
        '''
        fits the scaler
        Parameters:
            - X: x value data [numpy.array]
        Returns:
            - None
        '''
        ## make sure X is numpy.array
        X = np.array(X)
        self.get_min_max(X)
        self.is_fitted = True
        
    def fit_transform(self, X:np.array) -> np.array:
        '''
        fits the scaler and transforms the data
        Parameters:
            - X: x value data [numpy.array]
        Returns:
            - X_scaled: transformed data [numpy.array]
        '''
        self.fit(X)
        X_scaled = self.transform(X)
        return X_scaled
    
    def transform(self, X:np.array) -> np.array:
        '''
        transforms the data
        Parameters:
            - X: x value data [numpy.array]
        Returns:
            - X_scaled: transformed data [numpy.array]
        '''
        self.check_is_fitted()
        X_scaled = (X - self.min) / (self.max - self.min)
        return X_scaled
    
    def inverse_transform(self, X_scaled:np.array) -> np.array:
        '''
        retransforms the data
        Parameters:
            - X_scaled: scaled x value data [numpy.array]
        Returns:
            - X: retransformed data [numpy.array]
        '''
        self.check_is_fitted()
        X_rescaled = X_scaled * (self.max - self.min) + self.min
        return X_rescaled