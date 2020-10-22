'''
implements Deep Learning using Neural Networks
'''

## Imports
import numpy as np
from tqdm.notebook import tqdm
import plotly.offline as py
import plotly.graph_objs as go
from .preprocessing import train_test_split
from ._helper import convertSeconds
import abc, time
from .metrics import get_activation_function, loss_function, classifier_score
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_progress(metrics:dict, params:list, verbose) -> None:
    '''
    plots current loss curves and prints progress
    Parameters:
        - metrics: metrics to print and plot [Dictionary]
        - params: further parameters to use (current epoch and number of epochs) [List]
        - verbose: how detailed the train process shall be documented. Possible values are [Integer]
            - 0 -> no information (default)
            - 1 -> more detailed information
    Returns:
        - None
    '''
    (train_loss, train_metrics, test_loss, test_metrics) = metrics.values()
    (e, epochs, score, starttime, currenttime) = params
    epochtime = time.time() - currenttime
    clear_output(wait = True)
    length = 80 if verbose == 0 else 40
    progress = int(round((e * length - 1) / epochs, 0))
    if verbose > 0:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot()
        x = [i for i in range(train_loss.__len__())]
        ax.plot(x,train_loss, label="train-loss")
        ax.plot(x,test_loss, label="validation-loss")
        ax.set_title(f'Epoch {e+1}/{epochs}\n{"=" * (progress)}>{"_"*int(1.5*(length-progress))}\nTrain-Loss: {train_loss[-1]:.4f}; Train-{score.upper()}: {train_metrics[-1]:.4f}; Test-Loss: {test_loss[-1]:.4f}; Test-{score.upper()}: {test_metrics[-1]:.4f}\nEstimated time: {convertSeconds(currenttime-starttime)}<{convertSeconds(epochtime*(epochs-e))}, {(1/epochtime):.2f}it/s', loc="left")
        plt.legend()
        plt.show()
    else:
        print(f'Epoch {e+1}/{epochs}')
        print(f'{"=" * progress}>{"."*(length-progress-1)}')
        print(f'Train-Loss: {train_loss[-1]:.4f}; Train-{score.upper()}: {train_metrics[-1]:.4f}; Test-Loss: {test_loss[-1]:.4f}; Test-{score.upper()}: {test_metrics[-1]:.4f}')
        print(f'Estimated time: {convertSeconds(currenttime-starttime)}<{convertSeconds(epochtime*(epochs-e))}, {(1/epochtime):.2f}it/s')

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
            if count >= self.epochs:
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

class Layer(abc.ABC):
    '''
    abstract class for each kind of different layer. Each layer must be able to do two things
        - forward step -> output = layer.forward(input)
        - backpropagation -> grad_input = layer.backward(input, grad_output)
    some learn parameters -> updating them during layer.backward
    '''
    
    @abc.abstractmethod
    def __init__(self) -> None:
        '''
        force all layer to have an __init__ function to handle possible layer parameters
        Parameters:
            - None
        Initializes:
            - None
        Returns:
            - None
        '''
        pass
    
    @abc.abstractmethod
    def __name__(self, name:str = "Layer") -> str:
        '''
        forces all layers to have a __name__ to get to know the layers name
        Parameters:
            - name: desired name [String, default = "Layer"]
        Returns:
            - name: name of layer [String]
        '''
        return name

    @abc.abstractmethod
    def forward(self, input:np.array, *args) -> np.array:
        '''
        force all layer to have forward pass. Does forward step. Dummy layer's forward step just returns input
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - *args: for further Parameters, mostly unused
        Returns:
            - output: output of shape (batch_size, output_layer_len) [numpy.array]
        '''
        return input
    
    @abc.abstractmethod
    def backward(self, input:np.array, grad_output:np.array) -> np.array:       
        '''
        force all layer to have backward pass. Does backpropagation. Dummy layer's backpropagation returns grad_output. Perform chain rule:
        ## d_loss / d_x = (d_loss / d_layer) * (d_layer / d_x) ##
        --> only multiply grad_output with d_layer / d_x
        also update parameters (if layer has) using d_loss / d_layer
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - output: output of shape (batch_size, output_layer_len) [numpy.array]
        '''
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)
    
class ReLU(Layer):
    '''
    Rectified Unit Layer. Applies elementwise ReLU to all inputs. Converts all negative values to 0
    f(x) = max(0, input)
    '''
    
    def __init__(self) -> None:
        '''
        constructor of class
        Parameters:
            - None
        Initializes:
            - None
        Returns:
            - None
        '''
        pass

    def __name__(self) -> str:
        '''
        sets name of layer
        Parameters:
            - None
        Returns:
            - name: name of layer [String]
        '''
        return super().__name__("ReLU")
    
    def forward(self, input:np.array, *args) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - *args: unused
        Returns:
            - relu_forward: output of shape (batch_size, output_layer_len) [numpy.array]
            - ac_forward: output after activation function (batch_size, output_layer_len) [numpy.array] - same as relu_forward
        '''
        relu_forward = np.maximum(0, input)
        ac_forward = relu_forward.copy()
        return relu_forward, ac_forward
    
    def backward(self, input:np.array, grad_output:np.array) -> np.array:
        '''
        backward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - grad_input: output of shape (batch_size, output_layer_len) [numpy.array]
        '''
        relu_grad = input > 0
        grad_input = grad_output * relu_grad
        return grad_input
    
class Dense(Layer):
    '''
    Performs a learned affine transformation.
    f(x) = <W*x> + b
    '''
    
    def __init__(self, output_units:int, lr:float = 1e-3, activation_function:str = "sigmoid") -> None:
        '''
        constructor of class
        Parameters:
            - output_units: number of neurons for output (desired number of 'hidden_layers') [Integer]
            - lr: learning rate for backpropagation [Float, default = 1e-3]
            - activation_function: mode of the activation function. Possible values are [String]
                - Sigmoid-function --> "sigmoid"
                - Tangens hyperbolicus --> "tanh"
                - Rectified Linear Unit --> "relu"
                - Leaky Rectified Linear Unit --> "leaky-relu"
        Initializes:
            - lr
            - output_units
            - weights to be an empty numpy.array [numpy.array]
            - biases to be an array of Zeros [numpy.array]
            - faf: forward activation function [object]
            - baf: backward activation function [object]
        Returns:
            - None
        '''
        self.lr = lr
        self.output_units = output_units
        self.weights = np.array([])
        self.biases = np.zeros(output_units)
        self.faf = get_activation_function(activation_function)
        self.baf = get_activation_function(activation_function, True)

    def __name__(self) -> str:
        '''
        sets name of layer
        Parameters:
            - None
        Returns:
            - name: name of layer [String]
        '''
        return super().__name__("Dense")
    
    def forward(self, input:np.array, *args) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - *args: unused
        Returns:
            - dense_forward: output of shape (batch_size, output_layer_len) [numpy.array]
            - ac_forward: output after activation function (batch_size, output_layer_len) [numpy.array]
        '''
        ## make sure input is of type float
        input = input.astype(float)
        ## if weights are not properly initiated yet
        if self.weights.__len__() == 0:
            _, input_units = input.shape
            self.weights = np.random.normal(loc = 0.0, scale = np.sqrt( 2 / (input_units + self.output_units) ), size = (input_units, self.output_units))
        dense_forward = np.dot(input,self.weights) + self.biases
        ac_forward = self.faf(dense_forward)
        return dense_forward, ac_forward
    
    def backward(self, input:np.array, grad_output:np.array) -> np.array:
        '''
        backward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - grad_input: output of shape (batch_size, output_layer_len) [numpy.array]
        '''
        ## make sure input is of type float
        input = input.astype(float)
        ## get derivates of activation function
        derivates = self.baf(input)
        grad_input = np.dot(grad_output, self.weights.T) * derivates
        ## compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis = 0) * input.shape[0]
        ## check whether dimensions of new weights and biases are correct
        assert grad_weights.shape == self.weights.shape, "gradient of weights must match dimensionality of weights"
        assert grad_biases.shape == self.biases.shape, "gradient of biases must match dimensionality of biases"
        ## Update weights and biases
        self.weights = self.weights - self.lr * grad_weights
        self.biases = self.biases - self.lr * grad_biases
        return grad_input
    
class Convolution(Layer):
    '''
    Performs filtering on the given input using a learned filter (=pattern)
    '''
    
    def __init__(self, filters:int, kernel_size:tuple = (2,2), lr:float = 1e-3, activation_function:str = "sigmoid") -> None:
        '''
        constructor of class
        Parameters:
            - filters: number of filters to use [Integer]
            - kernel_size: Integer/Tuple, specifying height and width of the filters [Tuple, default = (2,2)]
            - lr: learning rate for backpropagation [Float, default = 1e-3]
            - activation_function: mode of the activation function. Possible values are [String]
                - Sigmoid-function --> "sigmoid"
                - Tangens hyperbolicus --> "tanh"
                - Rectified Linear Unit --> "relu"
                - Leaky Rectified Linear Unit --> "leaky-relu"
        Initializes:
            - lr
            - filters
            - kernel_size
            - weights: to be an emtpy array [numpy.array]
            - biases to be an array of Zeros [numpy.array]
            - faf: forward activation function [object]
            - baf: backward activation function [object][Integers]
        Returns:
            - None
        '''
        self.lr = lr
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weights = np.array([])
        self.biases = np.zeros((filters,1))
        self.faf = get_activation_function(activation_function)
        self.baf = get_activation_function(activation_function, True)

    def __name__(self) -> str:
        '''
        sets name of layer
        Parameters:
            - None
        Returns:
            - name: name of layer [String]
        '''
        return super().__name__("Convolution")
    
    def forward(self, input:np.array, *args) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - *args: unused
        Returns:
            - conv_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        '''
        ## make sure input is of type float
        input = input.astype(float)
        if self.weights.__len__() == 0:
            self.num_filt, self.filt_x, self.filt_y, self.filt_z = self.filters, *self.kernel_size, input.shape[-1]
            self.weights = np.random.normal(loc = 0, scale = 2/np.sqrt(np.prod((self.num_filt, self.filt_x, self.filt_y, self.filt_z))), size = (self.num_filt, self.filt_x, self.filt_y, self.filt_z))
        ## get image dimensions
        batch_size, imag_x, imag_y, imag_z = input.shape
        ## calc output dimensions
        out_x, out_y = (imag_x - self.filt_x) + 1, (imag_y - self.filt_y) + 1
        assert imag_z == self.filt_z, "depth of image must match depth of filters"
        ## init output
        conv_forward = np.zeros((batch_size, self.num_filt, out_x, out_y))
        ## convolve the filter (weights) over every part of the image, adding the bias at each step
        for batch in range(batch_size):
            for curr_f in range(self.num_filt):
                curr_x = 0
                while curr_x + self.filt_x <= imag_x:
                    curr_y = 0
                    while curr_y + self.filt_y <= imag_y:
                        conv_forward[batch, curr_f, curr_x, curr_y] = np.sum(self.weights[curr_f] * input[batch, curr_x:curr_x+self.filt_x, curr_y:curr_y+self.filt_y, :]) + self.biases[curr_f]
                        curr_y += 1
                    curr_x += 1
        conv_forward /= (out_x * out_y)
        ac_forward = self.faf(conv_forward)
        return conv_forward, ac_forward
    
    def backward(self, input:np.array, grad_output:np.array) -> np.array:
        '''
        backward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - grad_input: output of shape (batch_size, output_layer_len) [numpy.array]
        '''
        ## make sure input is of type float
        input = input.astype(float)
        ## get image dimensions
        batch_size, imag_x, imag_y, imag_z = input.shape
        ## init output
        grad_input = np.zeros(input.shape)
        grad_filters = np.zeros(self.weights.shape)
        grad_biases = np.zeros((self.num_filt, 1))
        ## get derivates of activation function
        derivates = self.baf(input)
        ## make backpropagation
        for batch in range(batch_size):
            for curr_f in range(self.num_filt):
                curr_x = 0
                while curr_x + self.filt_x <= imag_x:
                    curr_y = 0
                    while curr_y + self.filt_y <= imag_y:
                        ## loss gradient of weights (for weights update)
                        grad_filters[curr_f] += input[batch, curr_x:curr_x+self.filt_x, curr_y:curr_y+self.filt_y, :] * grad_output[batch, curr_f, curr_x, curr_y]
                        ## loss gradient of input to convolution operation
                        grad_input[batch, curr_x:curr_x+self.filt_x, curr_y:curr_y+self.filt_y, :] += grad_output[batch, curr_f, curr_x, curr_y] * self.weights[curr_f]
                        curr_y += 1
                    curr_x += 1
            ## loss gradient of bias
            grad_biases[curr_f] = np.sum(grad_output[:, curr_f])
        ## Update weights and biases
        self.weights = self.weights - self.lr * grad_filters
        self.biases = self.biases - self.lr * grad_biases
        return grad_input*derivates
    
class Pooling(Layer):
    '''
    Performs Pooling (Max or Average) on the given input
    '''
    
    def __init__(self, filter_size:tuple = (2,2), pooling_mode:str = "max") -> None:
        '''
        constructor of class
        Parameters:
            - filter_size: Integer/Tuple, specifying size of filter to shove over given image dimension (x,y) [Tuple, default = (2,2)]
            - pooling_mode: mode of pooling. Possible values are [String]
                - MaxPooling -> "max" (default)
                - AveragePooling -> "avg"
        Initializes:
            - x: first element of filter_size [Integer]
            - y: second element of filter_size [Integer]
            - pooling_mode
        Returns:
            - None
        '''
        self.x, self.y = filter_size if isinstance(filter_size, tuple) else (filter_size, filter_size)
        self.pooling_mode = pooling_mode

    def __name__(self) -> str:
        '''
        sets name of layer
        Parameters:
            - None
        Returns:
            - name: name of layer [String]
        '''
        return super().__name__("Pooling")

    def forward(self, input:np.array, *args) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - *args: unused
        Returns:
            - pool_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        '''
        batch_size, num_filt, imag_x, imag_y = input.shape
        ## calc output dimensions
        out_x, out_y = (imag_x - self.x) + 1, (imag_y - self.y) + 1
        pool_forward = np.zeros((batch_size, out_x, out_y, num_filt))
        ## perform pooling
        for batch in range(batch_size):
            for dim in range(num_filt):
                curr_x = 0
                while curr_x + self.x <= imag_x:
                    curr_y = 0
                    while curr_y + self.y <= imag_y:
                        if self.pooling_mode == "max":
                            pool_forward[batch, curr_x, curr_y, dim] = np.max(input[batch, dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                        elif self.pooling_mode == "avg":
                            pool_forward[batch, curr_x, curr_y, dim] = np.mean(input[batch, dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                        else:
                            print("this pooling mode is not implemented (yet)! Using AvgPool instead!")
                            pool_forward[batch, curr_x, curr_y, dim] = np.mean(input[batch, dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                        curr_y += 1
                    curr_x += 1
        ac_forward = pool_forward.copy()
        return pool_forward, ac_forward
    
    def backward(self, input:np.array, grad_output:np.array) -> np.array:
        '''
        backward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - grad_input: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        '''
        ## get image dimensions
        batch_size, num_filt, imag_x, imag_y = input.shape
        ## init output
        grad_input = np.zeros(input.shape)
        ## make backpropagation
        for batch in range(batch_size):
            for dim in range(num_filt):
                curr_x, out_x = 0, 0
                while curr_x + self.x <= imag_x:
                    curr_y, out_y = 0, 0
                    while curr_y + self.y <= imag_y:
                        if self.pooling_mode == "max":
                            ## obtain index of largest value in input for current window
                            idx = np.nanargmax(input[batch, dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                            (x, y) = np.unravel_index(idx, input[batch, dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y].shape)
                            ## set value to output
                            grad_input[batch, dim, out_x+x, out_y+y] = grad_output[batch, curr_x, curr_y, dim]
                        elif self.pooling_mode == "avg":
                            grad_input[batch, dim, out_x:out_x+self.x, out_y:out_y+self.x] = grad_output[batch, curr_x, curr_y, dim] * (1 / (self.x * self.y))
                        else:
                            print("this pooling mode is not implemented (yet)! Using AvgPool instead!")
                        
                        curr_y += self.y
                        out_y += 1
                    curr_x += self.x
                    out_x += 1
        return grad_input
    
class Flatten(Layer):
    '''
    performs the dimension reduction -> wrapping the input to dimension 1
    '''
    
    def __init__(self) -> None:
        '''
        constructor of class
        Initializes:
            - None
        Returns:
            - None
        '''

    def __name__(self) -> str:
        '''
        sets name of layer
        Parameters:
            - None
        Returns:
            - name: name of layer [String]
        '''
        return super().__name__("Flatten")
    
    def forward(self, input:np.array, *args) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - *args: unused
        Returns:
            - fullyconn_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, -1) [numpy.array]
        '''
        ## reshape data
        batch_size, (*_) = input.shape
        fullyconn_forward = input.reshape(batch_size, -1)
        ac_forward = fullyconn_forward.copy()
        return fullyconn_forward, ac_forward
    
    def backward(self, input:np.array, grad_output:np.array) -> np.array:
        '''
        backward step
        Parameters:
            - input: input of shape (batch_size, -1) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - grad_input: output of shape (batch_size, -1) [numpy.array]
        '''
        return grad_output.reshape(input.shape)

class Dropout(Layer):
    '''
    Performs filtering on the given input using a learned filter (=pattern)
    '''
    
    def __init__(self, factor:float) -> None:
        '''
        constructor of class
        Parameters:
            - factor: how much of the data shall be dropped - in range(0,1) [Float]
        Initializes:
            - factor
            - training: whether (=True) Layer is in use (only during training) or not (=False, default) [Boolean]
            - scale: how much the data gets rescaled (to keep sum of all data points while dropping some) [np.array]
        Returns:
            - None
        '''   
        ## make sure factor is in range(0,1)
        assert (factor < 1) & (factor > 0), "factor must be in between 0 and 1"
        self.factor = factor
        self.training = False
        self.scale = np.array([])
        
    def __name__(self) -> str:
        '''
        sets name of layer
        Parameters:
            - None
        Returns:
            - name: name of layer [String]
        '''
        return super().__name__("Dropout")
        
    def forward(self, input:np.array, *args) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - *args: whether Layer is in training (=True) mode or in prediction mode (=False) [Boolean]
        Returns:
            - drop_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        '''
        self.training, (*_) = args
        if self.training:
            ## get length of flattened input
            length = input.flatten().__len__()
            ## get as much zeros as factor * length
            zeros = np.zeros(int(length * self.factor))
            ## fill the rest with ones
            ones = np.ones(length - zeros.__len__())
            ## concat both, shuffle and reshape to input's shape
            scale = np.append(ones, zeros)
            np.random.shuffle(scale)
            self.scale = scale.reshape(input.shape)
            drop_forward = input * self.scale
        else:
            drop_forward = input.copy()
        ac_forward = drop_forward.copy()
        return drop_forward, ac_forward
    
    def backward(self, input:np.array, grad_output:np.array) -> np.array:
        '''
        backward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
            - grad_output: d_loss / d_layer [numpy.array]
        Returns:
            - grad_input: output of shape (batch_size, output_layer_len) [numpy.array]
        '''
        if self.training:
            grad_input = grad_output * self.scale
        else:
            grad_input = grad_output * input
        return grad_input
    
class NeuralNetwork(abc.ABC):
    '''
    abstract class for each kind of different neural network. Must be able to do three things
        - forward step -> output = layer.forward(input)
        - train -> train the network on given TrainData
        - predict -> predict a new sample
    '''
    
    @abc.abstractmethod
    def __init__(self, network:list) -> None:
        '''
        Force all neural networks to have an __init__() (=constructor) function
        constructor of class
        Parameters:
            - network: a list with all layers that the network shall contain [List]
        Initializes:
            - network
        Returns:
            - None
        '''
        self.network = network

    @abc.abstractmethod
    def __name__(self, name:str = "NeuralNetwork") -> str:
        '''
        forces all neural networks to have a __name__ to get to know the layers name
        Parameters:
            - name: desired name [String, default = "NeuralNetwork"]
        Returns:
            - name: name of NeuralNetwork [String]
        '''
        return name
        
    @abc.abstractmethod
    def check_scaled_data(self, data:np.array) -> None:
        '''
        Force all neural networks to have a check_scaled_data() function
        function that checks whether the given data is in range(0,1). Throws error if not
        Parameters:
            - data: data points to evaluate
        Returns:
            - None
        '''
        minimum = np.min(data)
        maximum = np.max(data)
        if (minimum < 0) or (maximum > 1):
            raise Exception("you have to scale the data into range(0,1). Network cannot work with unscaled data!")
    
    @abc.abstractmethod        
    def forward(self, X:np.array, training:bool = False) -> list:
        '''
        Force all neural networks to have a forward() function
        Compute activations of all network layers by applying them sequentially
        Parameters:
            - X: X data of shape (batch_size, input_layer_len) [numpy.array]
            - training: whether network gets trained (=True) or is in prediction mode (=False, default) [Boolean]
        Returns:
            - forwards: List of values for each layer [List]
            - activations: List of activations for each layer [List]
        '''
        ## init list of forwards, activations, set input for first layer
        forwards = []
        activations = []
        input = X
        ## Looping through all layer
        for l in self.network:
            f_s, input = l.forward(input, training)
            forwards.append(f_s)
            activations.append(input)
        ## make sure length of activations and forwards is same as number of layers
        assert len(activations) == len(self.network), "length of forward passed data must match length of layers"
        assert len(forwards) == len(self.network), "length of forward passed data must match length of layers"
        return forwards, activations
    
    @abc.abstractmethod    
    def train_step(self, X_train:np.array, y_train:np.array, loss_func:str) -> float:
        '''
        Force all neural networks to have a train_step() function
        performs networks training on given batches. Does nothing
        Parameters:
            - X_train: X_batch of shape (batch_size, sequence_length) [numpy.array]
            - y_train: corresponding y_batch of shape (batch_size, sequence_length) [numpy.array]
            - loss_func: mode of the loss function. Possible values are [String]
                - L1-norm Error (Regression) --> "l1"
                - L2-norm Error (Regression) --> "l2", (default)
                - Mean squared Error (Regression) --> "mse"
                - Mean absolute Error (Regression) --> "mae"
                - Root mean squared Error (Regression) --> "rmse"
                - Mean squared logarithmic Error (Regression) --> "mlse" 
                - Hinge (binary Classification) --> "hinge"
                - Squared Hinge (binary Classification) --> "squared-hinge"
                - Cross Entropy (binary Classification) --> "cross-entropy"
                - Categorical Cross Entropy (Multi-class Classification) --> "categorical-cross-entropy"
        Returns:
            - train_loss: trainings loss for current batch [Float]
        '''
        ## get layer activations
        layer_forwards, layer_activations = self.forward(X_train, True)
        ## layer[i] is the input for network[i]
        layer_inputs = [X_train] + layer_forwards
        ## prediction for this batch
        y_pred = layer_activations[-1]
        ## get last layer's error
        derivate = get_activation_function("sigmoid", True)(layer_inputs[-1])
        ## calculate loss
        loss = loss_function(y_train, y_pred, loss_func)
        loss_grad = loss_function(y_train, y_pred, loss_func, True) * derivate
        ## make backpropagation backwards through the network
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            ## update loss_grad, also updates the weights and biases
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)
        return loss
    
    @abc.abstractmethod       
    def train(self) -> None:
        '''
        Force all neural networks to have a train() function
        performs the training of the network for all steps (= epochs). Does nothing
        Parameters:
            - None
        Returns:
            - None
        '''
        pass   

    @abc.abstractmethod       
    def predict_step(self, X_test:np.array) -> np.array:
        '''
        Force all neural networks to have a predict_step() function
        does prediction of new data
        Parameters:
            - X_test: unknown sample [numpy.array]
        Returns:
            - y_pred: predicted 'regressed' data points [numpy.array]
        '''
        _, y_preds = self.forward(X_test)
        return y_preds[-1]
    
    @abc.abstractmethod       
    def predict(self) -> None:
        '''
        Force all neural networks to have a predict() function
        performs the prediction of x using the network. Does nothing
        Parameters:
            - None
        Returns:
            - None
        '''
        pass
    
class RegressorNetwork(NeuralNetwork):
    '''
    class for each kind of different regressor neural network. Must be able to do three things
        - forward step -> output = layer.forward(input)
        - train -> train the network on given TrainData
        - predict -> predict a new sample
    '''
    
    def __init__(self, network:list) -> None:
        '''
        constructor of class
        Parameters:
            - network: a list with all layers that the network shall contain [List]
        Initializes:
            - network
        Returns:
            - None
        '''
        super().__init__(network)

    def __name__(self) -> str:
        '''
        Parameters:
            - None
        Returns:
            - name: name of NeuralNetwork [String]
        '''
        return super().__name__("RegressorNetwork")
        
    def check_scaled_data(self, data:np.array) -> None:
        '''
        function that checks whether the given data is in range(0,1). Throws error if not
        Parameters:
            - data: data points to evaluate
        Returns:
            - None
        '''
        super().check_scaled_data(data)

    def forward(self, X:np.array, training:bool = False) -> list:
        '''
        Compute activations of all network layers by applying them sequentially
        Parameters:
            - X: X data of shape (batch_size, input_layer_len) [numpy.array]
            - training: whether network gets trained (=True) or is in prediction mode (=False, default) [Boolean]
        Returns:
            - forwards: List of values for each layer [List]
            - activations: List of activations for each layer [List]
        '''
        return super().forward(X, training)
    
    def train_step(self, X_train:np.array, y_train:np.array, loss_func:str) -> float:
        '''
        performs networks training on given batches
        Parameters:
            - X_train: X_batch of shape (batch_size, sequence_length) [numpy.array]
            - y_train: corresponding y_batch of shape (batch_size, sequence_length) [numpy.array]
            - loss_func: mode of the loss function. Possible values are [String]
                - L1-norm Loss --> "l1"
                - L2-norm Loss --> "l2", (default)
                - Mean squared Error --> "mse"
                - Mean absolute Error --> "mae"
                - Root mean squared Error --> "rmse"
                - Mean squared logarithmic Error --> "mlse" 
        Returns:
            - train_loss: trainings loss for current batch [Float]
        '''
        return super().train_step(X_train, y_train, loss_func)
        
    def iterate_minibatches(self, x:np.array, y:np.array, batch_size:int = 10, shuffle:bool = True) -> tuple:
        '''
        makes a list of minibatches
        Parameters:
            - x: x-values to train the network on [numpy.array]
            - y: y-values for loss calculation (= ground truth) [numpy.array]
            - batch_size: size of every batch [Integer, default = 10]
            - shuffle: whether (=True, default) or not (=False) to shuffle the data [Boolean]
        Returns:
            - Tuple of x_batch [numpy.array] and y_batch [numpy.array] [tuple]
        '''
        ## make sure x and y have same length
        assert len(x) == len(y), "x and y data streams must have same length"
        if shuffle:
            ## shuffle indize
            indices = np.random.permutation(len(x))
        for start_idx in range(0, len(x) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield x[excerpt], y[excerpt]
    
    def train(self, x:np.array, y:np.array, batch_size:int = 10, epochs:int = 100, loss_func:str = "l2", score:str = "l2", callbacks:list = [], verbose:int = 0) -> None:
        '''
        performs the training of the network for all steps (= epochs)
        Parameters:
            - x: x-values to train the network on [numpy.array]
            - y: y-values for loss calculation (= ground truth) [numpy.array]
            - batch_size: size of every batch [Integer, default = 10]
            - epochs: number of epochs to perform the training on [Integer, default = 100]
            - loss_func: mode of the loss function. Possible values are [String]
                - L1-norm Loss --> "l1"
                - L2-norm Loss --> "l2", (default)
                - Mean squared Error --> "mse"
                - Mean absolute Error --> "mae"
                - Root mean squared Error --> "rmse"
                - Mean squared logarithmic Error --> "mlse" 
            - score: mode of the scoring function. Possible values are [String]
                - L1-norm Loss --> "l1"
                - L2-norm Loss --> "l2", (default)
                - Mean squared Error --> "mse"
                - Mean absolute Error --> "mae"
                - Root mean squared Error --> "rmse"
                - Mean squared logarithmic Error --> "mlse" 
            - callbacks: list of Callback objects to use during training [List]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - None
        '''
        ## make sure x any y are floats
        x = x.astype(float)
        y = y.astype(float)
        if "Convolution" in [n.__name__() for n in self.network]:
            ## Convolution layer does scaling of data, so you don't need to check this
            pass
        else:
            ## check whether data is scaled into range(0,1)
            self.check_scaled_data(y)
        ## split datasets
        X_train, X_test, y_train, y_test = train_test_split(x, y)
        ## init train and test loss
        train_loss, test_loss, train_metrics, test_metrics = [], [], [], []  
        ## init the bar to show the progress
        starttime = time.time()
        for e in range(epochs): 
            ## init stopping and stop criterion arguments
            stop = 0
            stop_criterion = ""
            ## get current time
            epochtime = time.time()
            ## go through batches
            for x_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size, True):
                loss = self.train_step(x_batch, y_batch, loss_func)
            ## predict x train and x test
            train_pred = self.predict_step(X_train)
            test_pred = self.predict_step(X_test)
            ## update the loss and metric
            train_loss.append(np.mean(loss_function(y_train, train_pred, loss_func)))
            test_loss.append(np.mean(loss_function(y_test, test_pred, loss_func)))
            train_metrics.append(np.mean(loss_function(y_train, train_pred, score)))
            test_metrics.append(np.mean(loss_function(y_test, test_pred, score)))
            ## update progress of training
            metrics = {"train_loss": train_loss, "train_metrics": train_metrics, "test_loss": test_loss, "test_metrics": test_metrics}
            params = [e, epochs, score, starttime, epochtime]
            plot_progress(metrics, params, verbose)     
            for callback in callbacks:
                s, st = callback.evaluate(metrics)     
                stop += s
                stop_criterion += st
            ## check whether stopping criterion is reached
            if stop > 0:
                print(f'training aborted due to {", ".join(stop_criterion.split(" ")).strip()[:-1]}')
                break       
    
    def predict_step(self, X_test:np.array) -> np.array:
        '''
        does prediction of new data
        Parameters:
            - X_test: unknown sample [numpy.array]
        Returns:
            - y_pred: predicted 'regressed' data points [numpy.array]
        '''
        return super().predict_step(X_test)
    
    def predict(self, X_test:np.array, y_test:np.array, verbose:int = 0, markers:bool = False) -> np.array:
        '''
        performs the prediction of x using the network
        Parameters:
            - x_test: x-values to predict [numpy.array]
            - y_test: y-values for loss calculation (= ground truth) [numpy.array]
            - verbose: how detailed prediction shall be documented. Possible values are [Integer]
                - 0 -> no plot (default)
                - 1 -> show plot
            - markers: whether (=True) or not (=False, default) to use markers instead of a line as plot [Boolean]
        Returns:
            - y_pred: predicted y-values [numpy.array]
        '''
        ## get y_pred
        y_pred = self.predict_step(X_test)
        ## if shall be shown
        if verbose > 0:
            data = []
            if markers:
                data.append(go.Scatter(x=X_test.flatten(), y=y_pred.flatten(), mode="markers", marker_size=8, name="prediction"))
                data.append(go.Scatter(x=X_test.flatten(), y=y_test.flatten(), mode="markers", marker_size=8, name="ground truth"))
            else:
                data.append(go.Scatter(x=X_test.flatten(), y=y_pred.flatten(), name="prediction"))
                data.append(go.Scatter(x=X_test.flatten(), y=y_test.flatten(), name="ground truth"))
            fig = go.Figure(data)
            py.iplot(fig)
        return y_pred

class ClassifierNetwork(NeuralNetwork):
    '''
    class for each kind of different classifier neural network. Must be able to do three things
        - forward step -> output = layer.forward(input)
        - train -> train the network on given TrainData
        - predict -> predict a new sample
    '''
    
    def __init__(self, network:list) -> None:
        '''
        constructor of class
        Parameters:
            - network: a list with all layers that the network shall contain [List]
        Initializes:
            - network
        Returns:
            - None
        '''
        super().__init__(network)

    def __name__(self) -> str:
        '''
        Parameters:
            - None
        Returns:
            - name: name of NeuralNetwork [String]
        '''
        return super().__name__("ClassifierNetwork")
        
    def check_scaled_data(self, data:np.array) -> None:
        '''
        function that checks whether the given data is in range(0,1). Throws error if not
        Parameters:
            - data: data points to evaluate
        Returns:
            - None
        '''
        super().check_scaled_data(data)
    
    def forward(self, X:np.array, training:bool = False) -> list:
        '''
        Compute activations of all network layers by applying them sequentially
        Parameters:
            - X: X data of shape (batch_size, input_layer_len) [numpy.array]
            - training: whether network gets trained (=True) or is in prediction mode (=False, default) [Boolean]
        Returns:
            - forwards: List of values for each layer [List]
            - activations: List of activations for each layer [List]
        '''
        return super().forward(X, training)
    
    def train_step(self, X_train:np.array, y_train:np.array, loss_func:str) -> float:
        '''
        performs networks training on given batches
        Parameters:
            - X_train: X_batch of shape (batch_size, sequence_length) [numpy.array]
            - y_train: corresponding y_batch of shape (batch_size, sequence_length) [numpy.array]
            - loss_func: mode of the loss function. Possible values are [String]
                - Hinge --> "hinge"
                - Squared Hinge --> "squared-hinge"
                - Cross Entropy --> "cross-entropy"
                - Categorical Cross Entropy --> "categorical-cross-entropy"
        Returns:
            - train_loss: trainings loss for current batch [Float]
        '''
        return super().train_step(X_train, y_train, loss_func)
        
    def iterate_minibatches(self, x:np.array, y:np.array, batch_size:int = 10, shuffle:bool = True) -> tuple:
        '''
        makes a list of minibatches
        Parameters:
            - x: x-values to train the network on [numpy.array]
            - y: y-values for loss calculation (= ground truth) [numpy.array]
            - batch_size: size of every batch [Integer, default = 10]
            - shuffle: whether (=True, default) or not (=False) to shuffle the data [Boolean]
        Returns:
            - Tuple of x_batch [numpy.array] and y_batch [numpy.array] [tuple]
        '''
        ## make sure x and y have same length
        assert len(x) == len(y), "x and y data streams must have same length"
        if shuffle:
            ## shuffle indize
            indices = np.random.permutation(len(x))
        for start_idx in range(0, len(x) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield x[excerpt], y[excerpt]
    
    def train(self, x:np.array, y:np.array, batch_size:int = 10, epochs:int = 100, loss_func:str = "categorical-cross-entropy", score:str = "accuracy", callbacks:list = [], verbose:int = 0) -> None:
        '''
        performs the training of the network for all steps (= epochs)
        Parameters:
            - x: x-values to train the network on [numpy.array]
            - y: y-values for loss calculation (= ground truth) [numpy.array]
            - batch_size: size of every batch [Integer, default = 10]
            - epochs: number of epochs to perform the training on [Integer, default = 100]
            - loss_func: mode of the loss function. Possible values are [String]
                - Hinge (for binary classification) --> "hinge"
                - Squared Hinge (for binary classification) --> "squared-hinge"
                - Cross Entropy (for binary classification) --> "cross-entropy"
                - Categorical Cross Entropy (for multi-class classification) --> "categorical-cross-entropy", (default)
            - score: mode of the scoring function. Possible values are [String]
                - Recall --> "recall"
                - Precision --> "precision"
                - Accuracy --> "accuracy" = default
                - F1 --> "f1"
                - balanced Accuracy --> "balanced_accuracy"
            - callbacks: list of Callback objects to use during training [List]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - None
        '''
        y_ = np.zeros((y.shape[0],np.unique(y).__len__()))
        for i in range(y.__len__()):
            y_[i,y[i]] = 1
        y = y_.copy()
        ## make sure x any y are floats
        x = x.astype(float)
        y = y.astype(float)
        if "Convolution" in [n.__name__() for n in self.network]:
            ## Convolution layer does scaling of data, so you don't need to check this
            pass
        else:
            ## check whether data is scaled into range(0,1)
            self.check_scaled_data(x)
        ## split datasets
        X_train, X_test, y_train, y_test = train_test_split(x, y)
        ## init train and test loss
        train_loss, test_loss, train_metrics, test_metrics = [], [], [], []  
        ## init the bar to show the progress
        starttime = time.time()
        for e in range(epochs): 
            ## init stopping and stop criterion arguments
            stop = 0
            stop_criterion = ""
            ## get current time
            epochtime = time.time()
            ## go through batches
            for x_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size, True):
                loss = self.train_step(x_batch, y_batch, loss_func)
            ## predict x train and x test
            train_pred = self.predict_step(X_train)
            test_pred = self.predict_step(X_test)
            ## update the loss and metric
            train_loss.append(np.mean(loss_function(y_train, train_pred, loss_func)))
            test_loss.append(np.mean(loss_function(y_test, test_pred, loss_func)))
            train_metrics.append(np.mean(classifier_score(np.argmax(y_train, axis=-1), np.argmax(train_pred, axis=-1), score)))
            test_metrics.append(np.mean(classifier_score(np.argmax(y_test, axis=-1), np.argmax(test_pred, axis=-1), score)))
            ## update progress of training
            metrics = {"train_loss": train_loss, "train_metrics": train_metrics, "test_loss": test_loss, "test_metrics": test_metrics}
            params = [e, epochs, score, starttime, epochtime]
            plot_progress(metrics, params, verbose)   
            for callback in callbacks:
                s, st = callback.evaluate(metrics)     
                stop += s
                stop_criterion += st
            ## check whether stopping criterion is reached
            if stop > 0:
                print(f'training aborted due to {", ".join(stop_criterion.split(" ")).strip()[:-1]}')
                break          
            
    
    def predict_step(self, X_test:np.array) -> np.array:
        '''
        does prediction of new data
        Parameters:
            - X_test: unknown sample [numpy.array]
        Returns:
            - y_pred: predicted 'regressed' data points [numpy.array]
        '''
        return super().predict_step(X_test)
    
    def predict(self, X_test:np.array, y_test:np.array) -> tuple:
        '''
        performs the prediction of x using the network
        Parameters:
            - x_test: x-values to predict [numpy.array]
            - y_test: y-values for loss calculation (= ground truth) [numpy.array]
        Returns:
            - Tuple containing [Tuple]
               - label: predicted label [Integer]
               - proba: probability of label [Float]
        '''
        ## get y_pred
        y_pred = self.predict_step(X_test)
        label = np.argmax(y_pred)
        return label, y_pred[0,label]