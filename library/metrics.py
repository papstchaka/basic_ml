'''
implements different metrics for evaluation purposes and loss functions (deep_learning)
'''

## Imports
from autograd import elementwise_grad
import autograd.numpy as np

def get_activation_function(mode:str = "sigmoid", derivate:bool = False) -> object:
    '''
    returns corresponding activation function for given mode
    Parameters:
        - mode: mode of the activation function. Possible values are [String]
            - Sigmoid-function --> "sigmoid"
            - Tangens hyperbolicus --> "tanh"
            - Rectified Linear Unit --> "relu"
            - Leaky Rectified Linear Unit --> "leaky-relu"
        - derivate: whether (=True, default) or not (=False) to return the derivated value of given function and x [Boolean]
    Returns:
        - y: desired activation function [object]
    '''
    if mode == "sigmoid":
        y = lambda x: 1 / ( 1 + np.exp(-x) )
    elif mode == "tanh":
        y = lambda x: ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
    elif mode == "relu":
        y = lambda x: np.where(x <= 0, 0.0, 1.0) * x
    elif mode == "leaky-relu":
        y = lambda x: np.where(x <= 0, 0.1, 1.0) * x
    elif mode == "softmax":
        y = lambda x: np.exp(x-x.max()) / ( (np.exp(x-x.max()) / np.sum(np.exp(x-x.max()))) )
    else:
        print('Unknown activation function. linear is used')
        y = lambda x: x
    ## when derivation of function shall be returned
    if derivate:
        return elementwise_grad(y)
    return  y

def loss_function(y_test:np.array, y_pred:np.array, mode:str = "l2", derivate:bool = False) -> float:
    '''
    returns current loss for given x and trained weights and biases
    Parameters:
        - y_test: actual values - ground truth [numpy.array]
        - y_pred: predicted values [numpy.array]
        - mode: mode of the loss function. Possible values are [String]
            - L1-norm Error (for Regression) --> "l1"
            - L2-norm Error (for Regression) --> "l2" = default
            - Mean squared Error (for Regression) --> "mse"
            - Mean absolute Error (for Regression) --> "mae"
            - Root mean squared Error (for Regression) --> "rmse"
            - Mean squared logarithmic Error (for Regression) --> "mlse" 
            - Hinge (for binary classification) --> "hinge"
            - Squared Hinge (for binary classification) --> "squared-hinge"
            - Cross Entropy (for binary classification) --> "cross-entropy"
            - Categorical Cross Entropy (for multi-class classification) --> "categorical-cross-entropy"
        - derivate: whether (=True, default) or not (=False) to return the derivated value of given function and x [Boolean]
    Returns:
        - loss: calculated loss [Float]
    '''
    if mode == "l1": ## regression
        fx = lambda x,y: np.sum( np.abs(x - y), axis=-1)
    elif mode == "l2": ## regression
        fx = lambda x,y: np.sum(( x - y )**2, axis=-1)
    elif mode == "mse": ## regression
        fx = lambda x,y: np.sum(( x - y )**2, axis=-1 ) / x.__len__()
    elif mode == "mae": ## regression
        fx = lambda x,y: np.sum( np.abs(x - y), axis=-1 ) / x.__len__()
    elif mode == "rmse": ## regression
        fx = lambda x,y: np.sqrt(np.sum(( x - y )**2, axis=-1 ) / x.__len__())
    elif mode == "msle": ## regression
        fx = lambda x,y: np.sum( (np.log(x+1) - np.log(y+1))**2 , axis=-1) / x.__len__()
    elif mode == "hinge": ## binary classification
        fx = lambda x,y: np.sum(np.where((1 - x*y) <= 0, 0.0, 1.0) * (1 - x*y)) / x.__len__()
    elif mode == "squared-hinge": ## binary classification
        fx = lambda x,y: np.sum((np.where((1 - x*y) <= 0, 0.0, 1.0) * (1 - x*y) )**2) / x.__len__()
    elif mode == "cross-entropy": ## binary classification
        fx = lambda x,y: - np.sum(x * np.log(np.clip(y, 1e-7, 1-1e-7)) + (1-x) * np.log(np.clip(1-y, 1e-7, 1-1e-7))) / x.__len__()
    elif mode == "categorical-cross-entropy": ## multi-class classifaction
        fx = lambda x,y: - np.sum(x * np.log(np.clip(y, 1e-7, 1-1e-7))) / x.__len__()
    else:
        print('Unknown loss function. L2 is used')
        fx = lambda x,y: np.sum(( x - y )**2, axis=-1 )
    if derivate:
        return - elementwise_grad(fx)(y_test,y_pred)
    return fx(y_test,y_pred)

def calc_rates(y_test:np.array, y_pred:np.array) -> list:
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

def classifier_score(y_test:np.array, y_pred:np.array, mode:str = "accuracy", average:bool = True) -> float:
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
    assert(y_test.__len__() == y_pred.__len__()), "y_test and y_pred must have same length"
    ## make sure y_test and y_pred are actually numpy.arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    rates = calc_rates(y_test, y_pred)
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

def regressor_score(y_test:np.array, y_pred:np.array, mode:str = "l2") -> float:
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
    assert(y_test.__len__() == y_pred.__len__()), "y_test and y_pred must have same length"
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