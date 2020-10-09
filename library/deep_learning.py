'''
implements Deep Learning using Neural Networks
'''

from autograd import elementwise_grad
import autograd.numpy as np
from tqdm.notebook import tqdm
import plotly.offline as py
import plotly.graph_objs as go

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
        y = lambda x: x if x>0 else 0
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
        - mode: mode of the activation function. Possible values are [String]
            - L1-norm Loss --> "l1"
            - L2-norm Loss --> "l2" = default
            - Mean squared Error --> "mse"
            - Mean absolute Error --> "mae"
            - Root mean squared Error --> "rmse"
    Returns:
        - loss: calculated loss [Float]
    '''
    if mode == "l1":
        return np.sum( np.abs(x - y) )
    elif mode == "l2":
        return np.sum(( x - y )**2 )
    elif mode == "mse":
        return np.sum(( x - y )**2 ) / x.__len__()
    elif mode == "mae":
        return np.sum( np.abs(x - y) ) / x.__len__()
    elif mode == "rmse":
        return np.sqrt(np.sum(( x - y )**2 ) / x.__len__())
    elif mode == "cross-entropy": ## classification
        return - ( np.sum( x*np.log(y) + (1-x) * np.log(1-y) ) )
    else:
        print('Unknown loss function. L2 is used')
        return np.sum(( x - y )**2 )
        


class NeuralNetwork(object):
    '''
    class for implementation of a simple Neural Network, containing training function, feedforward and backward propagation
    '''
    
    def __init__(self, layers:list = [1, 100, 1], activations:list = []) -> None:
        '''
        constructor of class
        Parameters:
            - layers: to be the layers given by user [List of Integers, default = [1, 100, 1]]. construction is usually [input_layer, different amount of hidden layers, ..., output_layer]
            - activations: to be the desired activation functions [List of Strings, default = []]. If not set by user, will be initialized as "sigmoid"s for all layers
        initializes:
            - layers
            - activations - checks whether number of activation functions is same as number of layers
            - weights: to be a list filled with random values in the shape of current and following layer [List]
            - biases: to be a list filled with random values in the shape of current and following layer [List]
        Returns:
            - None
        '''
        ## init the activation functions if not set by user
        if activations.__len__() == 0:
            activations = ["sigmoid" for _ in range(layers.__len__()-1)]
        ## checks whether user provided enough activation functions
        assert(len(layers) == len(activations)+1)
        self.layers = layers
        self.activations = activations
        ## init biases and weights
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1], 1))
    
    def feedforward(self, x:np.array) -> tuple:
        '''
        feedforward function of the network, provides the forward step during each iteration
        Parameters:
            - x: x-values to train the network on = X_train [numpy.array]
        Returns:
            - tuple containing two values:
                - z_s: forwarded x-values [List of Floats]
                - a_s: activation values [List of Floats]
        '''
        ## copy sequence to make sure not to override the original data
        a = np.copy(x)
        ## init forwarded x values and activation values
        z_s = []
        a_s = [a]
        ## go through all the layers (excluding the output layer)
        for i in range(len(self.weights)):
            ## get desired activation function
            activation_function = get_activation_function(self.activations[i])
            ## calc current z and current a            
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)
    
    def backpropagation(self, y:np.array, z_s:list, a_s:list) -> tuple:
        '''
        backpropagation function of the network, provides the backward step during each iteration
        Parameters:
            - y: y values (=ground truth that shall be regressed) to train the network on = y_train [numpy.array]
            - z_s: forwarded x values [List of Floats]
            - a_s: activation values [List of Floats]
        Returns:
            - tuple containing two values:
                - dw: derivate of the weights (dA/dW) [List of Floats]
                - db: derivate of the biases (dA/dB) [List of Floats]
        '''
        ## init dA/dW -> activation_function with respect to weights
        dw = []
        ## init dA/dB -> activation_function with respect to biases
        db = []
        ## init dA/dZ -> activation_function with respect to 'error for each layer'
        deltas = [None] * len(self.weights)
        ## get last layer's error
        derivates = get_activation_function(self.activations[-1], True)(z_s[-1].flatten())
        deltas[-1] = ((y-a_s[-1])*derivates)
        ## perform backpropagation
        for i in reversed(range(len(deltas)-1)):
            ## get errors of all other layers
            derivates = get_activation_function(self.activations[i], True)(z_s[i]).flatten().reshape(z_s[i].shape)
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*derivates        
        ## get batch_size as being the shape of y
        batch_size = y.shape[1]
        ## get the derivates respect to weight matrix and biases
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
        return dw, db
        
    def train(self, x:np.array, y:np.array, batch_size:int = 10, epochs:int = 100, lr:float = 0.01) -> None:
        '''
        performs the training of the network for all steps (= epochs)
        Parameters:
            - x: x-values to train the network on [numpy.array]
            - y: y-values for loss calculation (= ground truth) [numpy.array]
            - batch_size: size of every batch [Integer, default = 10]
            - epochs: number of epochs to perform the training on [Integer, default = 100]
            - lr: learning rate for the gradient descent during new calculation of weights and biases [Float, default = 0.01 = 1e-2]
        Returns:
            - None
        '''
        ## init loss to be infinity
        loss = np.infty
        ## init the bar to show the progress
        pbar = tqdm(total = epochs)
        for e in range(epochs): 
            ## start with index 0 to get through all x- and y-values with batch size
            i = 0
            ## as long as end is not reached
            while(i < len(y)):
                ## get current x- and y-batch
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                ## update index
                i = i + batch_size
                ## perform feedforward step
                z_s, a_s = self.feedforward(x_batch)
                ## perform backpropagation step
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                ## update weights and biases
                self.weights = [w + lr * dweight for w,dweight in  zip(self.weights, dw)]
                self.biases = [w + lr * dbias for w,dbias in  zip(self.biases, db)]
            ## update the loss
            loss = loss_function(a_s[-1], y_batch)
            ## update progress of training
            pbar.set_description(f'Epoch: {e}; Loss: {loss}')
            pbar.update(1)
            
    def predict(self, x_test:np.array, y_test:np.array, verbose:int = 0) -> np.array:
        '''
        performs the prediction of x using the network
        Parameters:
            - x_test: x-values to predict [numpy.array]
            - y_test: y-values for loss calculation (= ground truth) [numpy.array]
            - verbose: how detailed prediction shall be documented. Possible values are [Integer]
                - 0 -> no plot (default)
                - 1 -> show plot
        Returns:
            - y_pred: predicted y-values [numpy.array]
        '''
        ## perform forward step
        _, a_s = self.feedforward(x_test)
        ## get y_pred
        y_pred = a_s[-1]
        ## if shall be shown
        if verbose > 0:
            data = []
            data.append(go.Scatter(x=x_test.flatten(), y=y_pred.flatten(), mode="markers", marker_size=8, name="prediction"))
            data.append(go.Scatter(x=x_test.flatten(), y=y_test.flatten(), mode="markers", marker_size=8, name="ground truth"))
            fig = go.Figure(data)
            py.iplot(fig)
        return y_pred