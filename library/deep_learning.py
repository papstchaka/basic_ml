'''
implements Deep Learning using Neural Networks
'''

## Imports
from autograd import elementwise_grad
import autograd.numpy as np
from tqdm.notebook import tqdm
import plotly.offline as py
import plotly.graph_objs as go
from .preprocessing import train_test_split
import abc

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
        y = lambda x: np.max(x,0)
    elif mode == "leaky-relu":
        y = lambda x: np.max([0.1*x,x])
    else:
        print('Unknown activation function. linear is used')
        y = lambda x: x
    ## when derivation of function shall be returned
    if derivate:
        return elementwise_grad(y)
    return  y

def loss_function(x:np.array, y:np.array, mode:str = "l2") -> float:
    '''
    returns current loss for given x and trained weights and biases
    Parameters:
        - x: x values to process [numpy.array]
        - y: y values that are wanted - "ground truth" [numpy.array]
        - mode: mode of the loss function. Possible values are [String]
            - L1-norm Loss --> "l1"
            - L2-norm Loss --> "l2" = default
            - Mean squared Error --> "mse"
            - Mean absolute Error --> "mae"
            - Root mean squared Error --> "rmse"
    Returns:
        - loss: calculated loss [Float]
    '''
    if mode == "l1":
        return np.sum( np.abs(x - y), axis=-1)
    elif mode == "l2":
        return np.sum(( x - y )**2, axis=-1)
    elif mode == "mse":
        return np.sum(( x - y )**2, axis=-1 ) / x.__len__()
    elif mode == "mae":
        return np.sum( np.abs(x - y), axis=-1 ) / x.__len__()
    elif mode == "rmse":
        return np.sqrt(np.sum(( x - y )**2, axis=-1 ) / x.__len__())
    elif mode == "cross-entropy": ## classification
        return - ( np.sum( x*np.log(y) + (1-x) * np.log(1-y) ) )
    else:
        print('Unknown loss function. L2 is used')
        return np.sum(( x - y )**2, axis=-1 )
        
class Layer(abc.ABC):
    '''
    abstract class for each kind of different layer. Each layer must be able to do two things
        - forward step -> output = layer.forward(input)
        - backpropagation -> grad_input = layer.backward(input, grad_output)
    some also learn parameters -> updating them during layer.backward
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
    def forward(self, input:np.array) -> np.array:
        '''
        force all layer to have forward pass. Does forward step. Dummy layer's forward step just returns input
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
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
    
    def forward(self, input:np.array) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
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
    
    def __init__(self, output_units:int, lr:float = 0.1, activation_function:str = "sigmoid") -> None:
        '''
        constructor of class
        Parameters:
            - output_units: number of neurons for output (desired number of 'hidden_layers') [Integer]
            - lr: learning rate for backpropagation [Float, default = 0.1]
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
    
    def forward(self, input:np.array) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, input_layer_len) [numpy.array]
        Returns:
            - dense_forward: output of shape (batch_size, output_layer_len) [numpy.array]
            - ac_forward: output after activation function (batch_size, output_layer_len) [numpy.array]
        '''
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
        ## get derivates of activation function
        derivates = self.baf(input)
        grad_input = np.dot(grad_output, self.weights.T) * derivates
        ## compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis = 0) * input.shape[0]
        ## check whether dimensions of new weights and biases are correct
        assert grad_weights.shape == self.weights.shape
        assert grad_biases.shape == self.biases.shape
        ## Update weights and biases
        self.weights = self.weights + self.lr * grad_weights
        self.biases = self.biases + self.lr * grad_biases
        return grad_input
    
class Convolution(Layer):
    '''
    Performs filtering on the given input using a learned filter (=pattern)
    '''
    
    def __init__(self, filters:int, kernel_size:tuple = (2,2,1), lr:float = 0.1, activation_function:str = "sigmoid") -> None:
        '''
        constructor of class
        Parameters:
            - filters: number of filters to use [Integer]
            - kernel_size: 2 Integers, specifying height, width and depth of the filters [Tuple, default = (2,2,1)]
            - lr: learning rate for backpropagation [Float, default = 0.1]
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
        self.kernel_size = kernel_size
        self.weights = np.array([])
        self.biases = np.zeros((filters,1))
        self.faf = get_activation_function(activation_function)
        self.baf = get_activation_function(activation_function, True)
    
    def forward(self, input:np.array) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        Returns:
            - conv_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        '''
        if self.weights.__len__() == 0:
            self.num_filt, self.filt_x, self.filt_y, self.filt_z = self.filters, *self.kernel_size
            self.weights = np.random.normal(loc = 0, scale = 2/np.sqrt(np.prod((self.num_filt, self.filt_x, self.filt_y, self.filt_z))), size = (self.num_filt, self.filt_x, self.filt_y, self.filt_z))
        ## get image dimensions
        imag_x, imag_y, imag_z = input.shape
        ## calc output dimensions
        out_x, out_y = (imag_x - self.filt_x) + 1, (imag_y - self.filt_y) + 1
        assert imag_z == self.filt_z
        ## init output
        conv_forward = np.zeros((self.num_filt, out_x, out_y))
        ## convolve the filter (weights) over every part of the image, adding the bias at each step
        for curr_f in range(self.num_filt):
            curr_x = 0
            while curr_x + self.filt_x <= imag_x:
                curr_y = 0
                while curr_y + self.filt_y <= imag_y:
                    conv_forward[curr_f, curr_x, curr_y] = np.sum(self.weights[curr_f] * input[curr_x:curr_x+self.filt_x, curr_y:curr_y+self.filt_y, :]) + self.biases[curr_f]
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
        ## get image dimensions
        imag_x, imag_y, imag_z = input.shape
        ## init output
        grad_input = np.zeros(input.shape)
        grad_filters = np.zeros(self.weights.shape)
        grad_biases = np.zeros((self.num_filt, 1))
        ## get derivates of activation function
        derivates = self.baf(input)
        ## make backpropagation
        for curr_f in range(self.num_filt):
            curr_x = 0
            while curr_x + self.filt_x <= imag_x:
                curr_y = 0
                while curr_y + self.filt_y <= imag_y:
                    ## loss gradient of weights (for weights update)
                    grad_filters[curr_f] += grad_output[curr_f, curr_x, curr_y] * input[curr_x:curr_x+self.filt_x, curr_y:curr_y+self.filt_y, :]
                    ## loss gradient of input to convolution operation
                    grad_input[curr_x:curr_x+self.filt_x, curr_y:curr_y+self.filt_y, :] += grad_output[curr_f, curr_x, curr_y] * self.weights[curr_f]
                    curr_y += 1
                curr_x += 1
            ## loss gradient of bias
            grad_biases[curr_f] = np.sum(grad_output[curr_f])
        ## Update weights and biases
        self.weights = self.weights + self.lr * grad_filters
        self.biases = self.biases + self.lr * grad_biases
        return grad_input*derivates
    
class Pooling(Layer):
    '''
    Performs Pooling (Max or Average) on the given input
    '''
    
    def __init__(self, filter_size:tuple = (2,2), pooling_mode:str = "max") -> None:
        '''
        constructor of class
        Parameters:
            - filter_size: size of filter to shove over given image dimension (x,y) [Tuple, default = (2,2)]
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
        self.x, self.y = filter_size
        self.pooling_mode = pooling_mode

    def forward(self, input:np.array) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        Returns:
            - pool_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        '''
        num_filt, imag_x, imag_y = input.shape
        ## calc output dimensions
        out_x, out_y = (imag_x - self.x) + 1, (imag_y - self.y) + 1
        pool_forward = np.zeros((out_x, out_y, num_filt))
        ## perform pooling
        for dim in range(num_filt):
            curr_x, out_x = 0, 0
            while curr_x <= imag_x:
                curr_y, out_y = 0, 0
                while curr_y <= imag_y:
                    if self.pooling_mode == "max":
                        pool_forward[out_x, out_y, dim] = np.max(input[dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                    elif self.pooling_mode == "avg":
                        pool_forward[out_x, out_y, dim] = np.mean(input[dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                    else:
                        print("this pooling mode is not implemented (yet)! Using MaxPool instead!")
                        pool_forward[out_x, out_y, dim] = np.max(input[dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                    curr_y += self.y
                    out_y += 1
                curr_x += self.x
                out_x += 1
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
        num_filt, imag_x, imag_y = input.shape
        ## init output
        grad_input = np.zeros(input.shape)
        ## make backpropagation
        for dim in range(num_filt):
            curr_x, out_x = 0, 0
            while curr_x + self.x <= imag_x:
                curr_y, out_y = 0, 0
                while curr_y + self.y <= imag_y:
                    ## obtain index of largest value in input for current window
                    idx = np.nanargmax(input[dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y])
                    (x, y) = np.unravel_index(idx, input[dim, curr_x:curr_x+self.x, curr_y:curr_y+self.y].shape)
                    ## set value to output
                    grad_input[dim, out_x+x, out_y+y] = grad_output[curr_x, curr_y, dim]
                    curr_y += self.y
                    out_y += 1
                curr_x += self.x
                out_x += 1
        return grad_input
    
class FullyConnected(Layer):
    '''
    performs the dimension reduction -> wrapping the input to dimension 1
    '''
    
    def __init__(self, batch_size:int) -> None:
        '''
        constructor of class
        Parameters:
            - batch_size: size of given batches
        Initializes:
            - batch_size
        Returns:
            - None
        '''
        self.batch_size = batch_size
    
    def forward(self, input:np.array) -> np.array:
        '''
        forward step
        Parameters:
            - input: input of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
        Returns:
            - fullyconn_forward: output of shape (batch_size, x_dim, y_dim, z_dim) [numpy.array]
            - ac_forward: output after activation function (batch_size, -1) [numpy.array]
        '''
        ## reshape data
        fullyconn_forward = input.reshape(self.batch_size, -1)
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
    
class NeuralNetwork():
    '''
    class for each kind of different neural network. Must be able to do three things
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
        self.network = network
        
    def check_scaled_data(self, data:np.array) -> None:
        '''
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
    
    def forward(self, X:np.array) -> list:
        '''
        Compute activations of all network layers by applying them sequentially
        Parameters:
            - X: X data of shape (batch_size, input_layer_len) [numpy.array]
        Returns:
            - activations: List of activations for each layer [List]
        '''
        ## init list of forwards, activations, set input for first layer
        forwards = []
        activations = []
        input = X
        ## Looping through all layer
        for l in self.network:
            f_s, input = l.forward(input)
            forwards.append(f_s)
            activations.append(input)
        ## make sure length of activations and forwards is same as number of layers
        assert len(activations) == len(self.network)
        assert len(forwards) == len(self.network)
        return forwards, activations
    
    def train_step(self, X_train:np.array, y_train:np.array) -> None:
        '''
        performs networks training on given batches
        Parameters:
            - X_train: X_batch of shape (batch_size, sequence_length) [numpy.array]
            - y_train: corresponding y_batch of shape (batch_size, sequence_length) [numpy.array]
        Returns:
            - None
        '''
        ## get layer activations
        layer_forwards, layer_activations = self.forward(X_train)
        ## layer[i] is the input for network[i]
        layer_inputs = [X_train] + layer_forwards
        ## prediction for this batch
        y_pred = layer_activations[-1]
        ## get last layer's error
        derivate = get_activation_function(derivate = True)(layer_inputs[-1])
        ## calculate loss
        loss_grad = (y_train - y_pred) * derivate
        ## make backpropagation backwards through the network
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            ## update loss_grad, also updates the weights and biases
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)
    
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
        assert len(x) == len(y)
        if shuffle:
            ## shuffle indize
            indices = np.random.permutation(len(x))
        for start_idx in range(0, len(x) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield x[excerpt], y[excerpt]
    
    def train(self, x:np.array, y:np.array, batch_size:int = 10, epochs:int = 100, loss_func:str = "l2") -> None:
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
        Returns:
            - None
        '''
        ## check whether data is scaled into range(0,1)
        self.check_scaled_data(y)
        ## split datasets
        X_train, X_test, y_train, y_test = train_test_split(x, y)
        ## init the bar to show the progress
        pbar = tqdm(total = epochs)
        for e in range(epochs): 
            ## go through batches
            for x_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size, True):
                self.train_step(x_batch, y_batch)
            ## predict x train and x test
            train_pred = self.predict_step(X_train)
            test_pred = self.predict_step(X_test)
            ## update the loss
            train_loss = loss_function(train_pred, y_train, loss_func)
            test_loss = loss_function(test_pred, y_test, loss_func)
            ## update progress of training
            pbar.set_description(f'Epoch: {e}; Train-Loss: {np.mean(train_loss)}; Test-Loss: {np.mean(test_loss)}')
            pbar.update(1)
            
    
    def predict_step(self, X_test:np.array) -> np.array:
        '''
        does prediction of new data
        Parameters:
            - X_test: unknown sample [numpy.array]
        Returns:
            - y_pred: predicted 'regressed' data points [numpy.array]
        '''
        _, y_preds = self.forward(X_test)
        return y_preds[-1]
    
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