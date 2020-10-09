'''
implements preprocessing for data
'''

import numpy as np

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
        Returns:
            - None
        '''
        self.min = None
        self.max = None
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
        if (self.min == None) or (self.max == None):
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