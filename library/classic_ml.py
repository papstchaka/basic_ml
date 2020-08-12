import numpy as np

class linear_regression:
    '''
    class that implements different approaches to do single dimensional or multidimensional linear regression:
        - single dimensional approaches:
            - f(x) = w*x as direct approach
            - f(x) = w0 + w1*x as direct approach
            - f(x) = w*x with gradient descent approach
        - multidimensional approaches:
            - f(x) = W dot x using direct approach
    '''
    
    def __init__(self):
        '''
        constructor of class
        initializes:
            - w to be an empty array
            - dim (dimension) to be 0
        '''
        self.w = np.array([])
        self.dim = 0
    
    def single_dimension(self, x:np.array, y:np.array, mode:str = "gradient_descent") -> None:
        '''
        responsible to do the desired single dimensional approach
        Parameters:
            - x: X Data to train on [numpy.array]
            - y: Y Data to train on [numpy.array]
            - mode: desired mode - possible ones are [String]:
                - direct f(x) = w*x --> "simple"
                - gradient descent f(x) = w*x --> "gradient_descent" (default)
                - direct f(x) = w0 + w1*x --> "complex"
        Returns:
            - None
        '''
        ## check mode
        if mode == "simple":
            w = ( sum([x[i] * y[i] for i,_ in enumerate(x)]) / sum([x_i**2 for x_i in x]) )
            self.w = np.array([0, w])
        elif mode == "gradient_descent":
            self.gradient_descent_regression(x, y)
        elif mode == "complex":
            w0 = ( ( sum(y) * sum([x_i**2 for x_i in x]) - sum(x) * sum([x[i] * y[i] for i,_ in enumerate(x)]) ) / ( len(x) * sum([x_i**2 for x_i in x]) - sum(x)**2 ) )
            w1 = ( ( len(x) * sum([x[i] * y[i] for i,_ in enumerate(x)]) - sum(x) * sum(y) ) / ( len(x) * sum([x_i**2 for x_i in x]) - sum(x)**2 ) )
            self.w = np.array([w0, w1])
            
    def multi_dimension(self, x:np.array, y:np.array) -> None:
        '''
        responsible to do the desired mutlti dimensional approach
        Parameters:
            - x: X Data to train on [numpy.array]
            - y: Y Data to train on [numpy.array]
        Returns:
            - None
        '''
        w_star = ( np.linalg.inv(x.T * x) * x.T * y )
        self.w = w_star
        
    def gradient_descent_regression(self, x:np.array, y:np.array, w:int = 0, learning_rate:float = 0.1, border:float = 0.01, batch_size:int = 5, break_point:int = 1000) -> None:        
        '''
        Implementation of single dimensional gradient descent
        Parameters:
            - x: X Data to train on [numpy.array]
            - y: Y Data to train on [numpy.array]
            - w: initial w to start with [Integer, default = 0]
            - learning_rate: Learning rate to use [Float, default = 0.1]
            - border: Accuracy between w_(t) and w_(t-1), when to stop the iteration because of convergence[Float, default = 0.01]
            - batch_size: Batch size to use [Integer, default = 5]
            - break_point: Accuracy between w_(t) and w_(t-1), when to stop the iteration because of divergence (then w=0) [Integer, default = 1000]
        Returns:
            - None
        '''
        ## set batch size
        if batch_size >= len(x) or batch_size == 0:
            batch_size = len(x)
        ## get number of samples per batch
        batches = int(np.ceil(len(x) / batch_size))
        ## start iteration
        while True:
            old_w = w
            for j in range(batches):
                w = w - ( ( learning_rate / batch_size ) * 
                         sum([(w * x[i] - y[i]) * x[i] for i,_ in enumerate(x[j*batch_size : j*batch_size + batch_size])]) )
            ## break because of convergence
            if np.abs(old_w - w) <= border:
                break
            ## break because of divergence
            if np.abs(old_w - w) >= break_point:
                w = 0
                print('gradient descend failed because w diverges')
                break
        self.w = np.array([0, w])
            
            
    def train(self, x:np.array, y:np.array, mode:str = "simple") -> None:
        '''
        training function
        Parameters:
            - x: X Data to train on [numpy.array]
            - y: Y Data to train on [numpy.array]
            - mode: desired mode - possible ones are [String]:
                - direct f(x) = w*x --> "simple" (default)
                - gradient descent f(x) = w*x --> "gradient_descent" 
                - direct f(x) = w0 + w1*x --> "complex"
        Returns:
            - None
        '''
        ## be sure x and y are arrays with the right shapes
        x = np.array(x)
        x = x if len(x.shape)>1 else x.reshape(-1,1)
        self.dim = x.shape[1]
        y = np.array(y).reshape(-1,self.dim)
        
        if x.shape[1] < 2:
            self.single_dimension(x, y, mode)
        else:
            self.multi_dimension(x, y)
    
    def predict(self, x_test:np.array) -> np.array:
        '''
        prediction function
        Parameters:
            - x_test: X Data to predict [numpy.array]
        Returns:
            - y_test: predicted Y Data [numpy.array]
        '''
        ## be sure x_test is array and in the right shape
        x_test = np.array(x_test).reshape(-1,self.dim)
        
        if self.dim < 2:
            y_test = np.array([self.w[0] + self.w[1] * x_i for x_i in x_test])
        else:
            y_test = np.array([self.w.dot(x_i) for x_i in x_test])
            
        return y_test
            