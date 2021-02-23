'''
class that implements gaussian processes as regression algorithm 
'''

## Imports
import numpy as np
from ..utils._helper import _regressor
    
class gp(_regressor):
    
    def __init__(self):
        '''
        constructor of class - imports needed modules
        initializes:
            - K: kernel matrix of x-values [numpy.array]
            - K*: kernel matrix of x/x_test values [numpy.array]
            - k_2star: kernel matrix of x_test values [numpy.array]
            - sigma: noise constant to add [Float]
        Returns:
            - None
        '''
        self._imports()
        self.k = np.array([])
        self.k_star = np.array([])
        self.k_2star = np.array([])
        self.sigma = 0.0
        
    def _imports(self) -> None:
        '''
        handles the imports for the class
        Parameters:
            - None
        Returns:
            - None
        '''
        global np, itertools
        import numpy, itertools
        np, itertools = numpy, itertools
    
    def kernel_function(self, x1:float, x2:float, sigma:float, l:float, mode:str, training:bool = False) -> float:
        '''
        return the K Matrix of the corresponding kernel function using given data points
        Parameters:
            - x1: data point of first set [Float]
            - x2: data point of second set [Float]
            - sigma: noise constant to add [Float]
            - l: lenghtscale of GP [Float]
            - mode: which kind of kernel to use. Possible values are [String]
                - rbf: Radial Basis Function
            - training: whether training (= True) mode or prediction mode (= False) [Boolean, default = False]
        Returns:
            - kernel_val: Value of used Kernel Function [Float]
        '''
        noise = 0
        ## when training and only for equal data points
        if training and (x1 == x2):
            noise = sigma
        if mode == "rbf":
            return np.exp(- (np.linalg.norm(x1 - x2)**2 / (2 * l**2))) + noise
        return np.nan
    
    def calc_kernel(self, x:np.array, x_star:np.array, sigma:float, l:float, mode:str) -> tuple:
        '''
        calculates the kernel matrizes K, K* and K** given training and test data
        Parameters:
            - x: training data points [numpy.array]
            - x_star: testing data points [numpy.array]
            - sigma: noise constant to add [Float]
            - l: lenghtscale of GP [Float]
            - mode: which kind of kernel to use. Possible values are [String]
                - rbf: Radial Basis Function
        Returns:
            - (K, K*, K**): tuple of calculated kernel matrizes [tuple]
        '''
        ## calc K and reshape it
        k = np.array([self.kernel_function(i,j, sigma, l, mode, True) for (i,j) in itertools.product(x,x)]).reshape(len(x),len(x))
        ## calc K* and reshape it
        k_star = np.array([self.kernel_function(i,j, sigma, l, mode) for (i,j) in itertools.product(x_star,x)]).reshape(len(x_star),len(x))
        ## calc K** and reshape it
        k_2star = np.array([self.kernel_function(i,j, sigma, l, mode) for (i,j) in itertools.product(x_star,x_star)]).reshape(len(x_star),len(x_star))
        return (k, k_star, k_2star)
    
    def train(self, x:np.array, x_test:np.array, sigma:float, l:float, mode:str = "rbf") -> tuple:
        '''
        training function for Gaussian Process given x and y data. Also predicts the regressed value(s) for given x_test
        Parameters:
            - x: training data points [numpy.array]
            - x_test: testing data points [numpy.array]
            - sigma: noise constant to add [Float]
            - l: lenghtscale of GP [Float]
            - mode: which kind of kernel to use. Possible values are [String]
                - rbf: Radial Basis Function (default)
        Returns:
            - None
        '''
        x = super().train(x)
        ## make sure x_test are numpy.arrays
        x_test = np.array(x_test)
        ## calc all kernel matrizes
        self.k, self.k_star, self.k_2star = self.calc_kernel(x,x_test,sigma,l,mode)
        ## keep track of sigma
        self.sigma = sigma
        
    def predict(self, y:np.array, return_cov=False) -> tuple:
        '''
        prediction function for Gaussian Process
        Parameters:
            - y: training labels [numpy.array]
            - return_cov: whether (= True) or not (= False) return the covariance of GP [Boolean, default=False]
        Returns:
            if return_cov == True:
                - (y_pred, covariance): tuple of regressed y_pred and covariance [tuple]
            else:
                - y_pred: regressed y_pred [numpy.array]
        '''
        y = super().predict(y)
        ## get shape of x
        n = self.k.shape[0]
        ## get regressed y_pred
        y_pred = np.dot(self.k_star, np.dot(np.linalg.inv(self.k + (self.sigma)*np.eye(n)), (y.reshape([n, 1]))))
        ## calc covariance
        if return_cov:
            covariance = self.k_2star - np.dot(self.k_star, np.dot(np.linalg.inv(self.k + (self.sigma)*np.eye(n)), self.k_star.T))
            return (y_pred, covariance)
        return y_pred
    
    def score(self, y_test:np.array, y_pred:np.array, mode:str = "l2") -> float:
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
        return super().score(y_test, y_pred, mode)