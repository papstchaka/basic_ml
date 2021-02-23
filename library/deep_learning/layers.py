'''
implements different kinds of layers for a Neural Network
'''

## Imports
import numpy as np
import abc
from ..utils.metrics import get_activation_function

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