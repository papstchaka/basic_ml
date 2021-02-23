'''
implements to kinds of Neural Networks for Deep Learning
'''

## Imports
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import abc, time
from .utils import plot_progress
from ..utils.preprocessing import train_test_split
from ..utils.metrics import get_activation_function, loss_function, classifier_score
    
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
                - Huber (for Regression) --> "huber"
                - Hinge (binary Classification) --> "hinge"
                - Squared Hinge (binary Classification) --> "squared-hinge"
                - Cross Entropy (binary Classification) --> "cross-entropy"
                - Categorical Cross Entropy (Multi-class Classification) --> "categorical-cross-entropy"
                - Kullback-Leibler Divergence (for multi-class classification) --> "kullback-leibler"
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
                - Huber (for Regression) --> "huber"
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
                - Kullback-Leibler Divergence (for multi-class classification) --> "kullback-leibler"
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