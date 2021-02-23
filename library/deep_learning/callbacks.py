'''
implements different kinds of callbacks for deep learning
'''

## Imports
import numpy as np
import abc

class Callback(abc.ABC):
    '''
    abstract class for each kind of different callback
    '''
    
    def __init__(self, function:str, validation:bool = True, epochs:int = 10, tolerance:float = 1e-2, **kwargs) -> None:
        '''
        force all layer to have an __init__ function to handle possible layer parameters
        Parameters:
            - function: metric to apply callback to - for example "loss" or "scoring" [String]
            - validation: whether (=True, default) or not (=False) evaluate callback on validation set (otherwise on training set) [Boolean]
            - epochs: number of epochs to wait until callback gets activated [Integer, default = 10]
            - tolerance: tolerance in change providing the callback to be activated [Float, default = 1e-4]
            - **kwargs: enabling user to provide a dictionary with all needed information, no matter of the order [Dictionary]
        Initializes:
            - function
            - validation to be either train or test [String]
            - epochs
            - tolerance
        Returns:
            - None
        '''
        self.function = function
        self.validation = "test" if validation else "train"
        self.epochs = epochs
        self.tolerance = tolerance
    
    @abc.abstractmethod
    def __name__(self, name:str = "Callback") -> str:
        '''
        forces all layers to have a __name__ to get to know the callbacks name
        Parameters:
            - name: desired name [String, default = "Callback"]
        Returns:
            - name: name of callback [String]
        '''
        return name

    def earlystopping(self, data:list) -> int:
        '''
        implements the earlystopping function
        Parameters:
            - data: data of desired metric [List]
        Returns:
            - stop: whether (=0) or not (=1) to stop the training [Integer]
        '''
        ## init stop criterion
        stop = 0
        ## init the changes
        changes = np.ones((self.epochs))
        ## go through last 'epochs' elements of data
        for i, element in enumerate(data[-self.epochs:]):
            changes[i] = np.abs(element - data[-self.epochs+i+1])
        ## check if the mean of the changes is below tolerance
        if changes.mean() <= self.tolerance:
            stop  = 1
        return stop
    
    def stopoverfitting(self, data:list) -> int:
        '''
        implements the stopoverfitting function
        Parameters:
            - data: data of desired metric [List]
        Returns:
            - stop: whether (=0) or not (=1) to stop the training [Integer]
        '''
        ## init stop criterion and counter
        stop = 0
        count = 0
        ## go through last 'epochs' elements of data
        for i, element in enumerate(data[-self.epochs:]):
            ## check whether next element in data is bigger or equal than previous (+ tolerance)
            if element + self.tolerance <= data[-self.epochs+i+1]:
                count += 1
            else:
                count = 0
            if count >= self.epochs - 1:
                stop = 1
                break
        return stop
    
    @abc.abstractmethod
    def evaluate(self, metrics:dict, callback:str) -> int:
        '''
        forces all callbacks to have a evaluate() function that checks whether callback should get active. Does nothing
        Parameters:
            - metrics: Dictionary with all metrics that are collected during traing (trainloss, trainscore, testloss, testscore) [Dictionary]
            - callback: desired callback to use [String]
        Returns:
            - stop: whether (=0) or not (=1) to stop the training [Integer]
        '''
        ## init stop criterion
        stop = 0
        ## check whether choosen metric is available
        possible_metrics = [key.split("_")[1] for key in metrics.keys()]
        if self.function in possible_metrics:
            data = metrics[f'{self.validation}_{self.function}']
        else:
            print(f"this metric is not available! Using {list(metrics.keys())[0]} instead!")
            data = metrics[list(metrics.keys())[0]]
        ## only perform callback if more than the desired number of epochs was actually already done
        if data.__len__() >= self.epochs:
            if callback == "EarlyStopping":
                stop = self.earlystopping(data)
            elif callback == "StopOverfitting":
                stop = self.stopoverfitting(data)
        return stop        
        
class EarlyStopping(Callback):
    '''
    Callback that stops training procedure if model does not perform any (noteworthy) changes in given function
    '''
    
    def __init__(self, function:str, validation:bool = True, epochs:int = 10, tolerance:float = 1e-2, **kwargs) -> None:
        '''
        constructor of class
        Parameters:
            - function: metric to apply callback to - for example "loss" or "scoring" [String]
            - validation: whether (=True, default) or not (=False) evaluate callback on validation set (otherwise on training set) [Boolean]
            - epochs: number of epochs to wait until callback gets activated [Integer, default = 10]
            - tolerance: tolerance in change providing the callback to be activated [Float, default = 1e-4]
            - **kwargs: enabling user to provide a dictionary with all needed information, no matter of the order [Dictionary]
        Initializes:
            - function
            - validation
            - epochs
            - tolerance
        Returns:
            - None
        '''
        super().__init__(function, validation, epochs, tolerance, **kwargs)
        
    def __name__(self) -> str:
        '''
        sets name of callback
        Parameters:
            - None
        Returns:
            - name: name of callback [String]
        '''
        return super().__name__("EarlyStopping")
    
    def evaluate(self, metrics:dict) -> int:
        '''
        checks whether callback should get active
        Parameters:
            - metrics: Dictionary with all metrics that are collected during traing (trainloss, trainscore, testloss, testscore) [Dictionary]
        Returns:
            - Tuple containing [Tuple]  
                - stop: whether (=0) or not (=1) to stop the training [Integer]
                - stop_criterion: which callback was responsible for aborting [String]
        '''
        ## init stop_criterion
        stop_criterion = ""
        ## evaluate
        stop = super().evaluate(metrics, "EarlyStopping")
        if stop > 0:
            stop_criterion = "EarlyStopping "
        return stop, stop_criterion

class StopOverfitting(Callback):
    '''
    Callback that stops training procedure if model starts to overfit in given function
    '''
    
    def __init__(self, function:str, epochs:int = 10, tolerance:float = 1e-2, **kwargs) -> None:
        '''
        constructor of class
        Parameters:
            - function: metric to apply callback to - for example "loss" or "scoring" [String]
            - epochs: number of epochs to wait until callback gets activated [Integer, default = 10]
            - tolerance: tolerance in change providing the callback to be activated [Float, default = 1e-4]
            - **kwargs: enabling user to provide a dictionary with all needed information, no matter of the order [Dictionary]
        Initializes:
            - function
            - validation to be always set to True (prooving overfitting on train metrics does not make sense)
            - epochs
            - tolerance
        Returns:
            - None
        '''
        super().__init__(function, True, epochs, tolerance, **kwargs)
        
    def __name__(self) -> str:
        '''
        sets name of callback
        Parameters:
            - None
        Returns:
            - name: name of callback [String]
        '''
        return super().__name__("StopOverfitting")
    
    def evaluate(self, metrics:dict) -> int:
        '''
        checks whether callback should get active
        Parameters:
            - metrics: Dictionary with all metrics that are collected during traing (trainloss, trainscore, testloss, testscore) [Dictionary]
        Returns:
            - Tuple containing [Tuple]  
                - stop: whether (=0) or not (=1) to stop the training [Integer]
                - stop_criterion: which callback was responsible for aborting [String]
        '''
        ## init stop_criterion
        stop_criterion = ""
        ## evaluate
        stop = super().evaluate(metrics, "StopOverfitting")
        if stop > 0:
            stop_criterion = "StopOverfitting "
        return stop, stop_criterion