'''
class that implements different approaches of dimensionality reduction:
    - LDA: Linear Discriminant Analysis
    - PCA: Principal Component Analysis
    - FastICA: Fast Independent Component Analysis
'''

## Imports
import numpy as np

class dimension_reduction:
    
    def __init__(self) -> None:
        '''
        constructor of class
        initializes:
            - is_fitted: whether scaler is fitted or not to False [Boolean]
            - X as Train Data to emtpy numpy array
        Returns:
            - None
        '''
        self._imports()
        self.is_fitted = False
        self.X = np.array([])
        
    def _imports(self) -> None:
        '''
        handles the imports for the class - imports needed modules
        Parameters:
            - None
        Returns:
            - None
        '''
        global np, elementwise_grad, npa
        import autograd.numpy, numpy
        from autograd import elementwise_grad
        np, elementwise_grad, npa = numpy, elementwise_grad, autograd.numpy
        
    def check_is_fitted(self) -> None:
        '''
        check whether model is already fitted
        '''
        if self.is_fitted:
            pass
        else:
            raise Exception("model is not fitted yet!")
    
    def calc_mean(self, X:np.array, verbose:int) -> np.array:
        '''
        calculates the means of the classes
        Parameters:
            - X: Train Data [numpy.array]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - means: calculated means [numpy.array]
        '''
        means = np.mean(X,axis=0)
        if verbose>0:
            print(f'Mean of data:\n{means}')
        return means
    
    def calc_sw(self, means:np.array, verbose:int) -> np.array:
        '''
        calculates scatter matrix S_W of the classes with given means
        Parameters:
            - means: means of classes [numpy.array]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - S_W: calculated scatter matrix [numpy.array]
        '''
        if len(self.X.shape)>2:
            S_W = np.zeros((len(self.X),len(self.X)))
            for i,class_data in enumerate(self.X):
                ## scatter matrix for every class
                class_sc_mat = np.zeros((len(self.X),len(self.X)))
                for point in class_data:
                    ## make column vector
                    point, cv = point.reshape(len(self.X),1), means[i].reshape(len(self.X),1)
                    class_sc_mat += (point-cv).dot((point-cv).T)
                ## sum up all scatter matrizes
                S_W += class_sc_mat
        else:
            S_W = np.zeros((2,2))
            for point in self.X:
                ## make column vector
                point, cv = point.reshape(-1,1), means.reshape(-1,1)
                S_W += (point-cv).dot((point-cv).T)
        if verbose>0:
            print(f'within-class Scatter Matrix:\n{S_W}')
        return S_W
    
    def calc_cov(self, verbose:int) -> np.array:
        '''
        calculates covariance of Train Data
        Parameters:
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - cov: calculate covariance matrix [numpy.array]
        '''
        X = [x.reshape(1,-1) for x in self.X]
        cov = (1 / len(self.X)) * sum( [x.T * x for x in X] )
        if verbose>0:
            print(f'Covariance Matrix:\n{cov}')
        return cov
    
    def calc_eig_vals(self, S_W:np.array, S_B:np.array, verbose:int) -> list: 
        '''
        calculates (eigenvalue, eigenvector)-pairs for given scatter- and covariance matrix
        Parameters:
            - S_W: given scatter matrix [numpy.array]
            - S_B: given covariance matrix [numpy.array]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - eig_pairs: sorted list with calculated (eigenvalue, eigenvector pairs) - descending [numpy.array]
        '''
        ## get eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        if verbose>0:
            for i in range(len(eig_vals)):
                eigv = eig_vecs[:,i].reshape(len(S_B),1)
                np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                                     eig_vals[i] * eigv,
                                                     decimal=6, err_msg='', verbose=True)
            print('eigenvalues correct')
            # Visually confirm that the list is correctly sorted by decreasing eigenvalues
            print('Variance explained:')
            eigv_sum = sum(eig_vals)
            for i,j in enumerate(eig_pairs):
                print(f'eigenvalue {i+1}: {(j[0]/eigv_sum).real:.2f}%')
        return eig_pairs
    
    def calc_W(self, eig_pairs:np.array, dim:int, verbose:int) -> np.array:
        '''
        calculate transforming vector for pca given (eigenvalue, eigenvector)-pairs
        Parameters:
            - eig_pairs: sorted list with calculated (eigenvalue, eigenvector pairs) - descending [numpy.array]
            - dim: desired dimension of projected data [Integer]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        '''
        W = np.hstack([eig_pairs[i][1].reshape(len(eig_pairs),1) for i in range(dim)])
        if verbose>0:
            print(f'Matrix W:\n{W.real}')
        return W
    
    def f(self, mode:str = "tanh", derivate:bool = False) -> object:
        '''
        returns corresponding function for given mode
        Parameters:
            - mode: mode of the function. Possible values are [String]
                - Tangens hyperbolicus --> "tanh"
            - derivate: whether (=True, default) or not (=False) to return the derivated value of given function and x [Boolean]
        Returns:
            - y: desired activation function [object]
        '''
        if mode == "tanh":
            y = lambda x: ( npa.exp(x) - npa.exp(-x) ) / ( npa.exp(x) + npa.exp(-x) )
        elif mode == "exp":
            y = lambda x: x * npa.exp(- (x ** 2) / 2)
        elif mode == "cube":
            y = lambda x: x**3
        else:
            print('Unknown activation function. tanh is used')
            y = lambda x: ( npa.exp(x) - npa.exp(-x) ) / ( npa.exp(x) + npa.exp(-x) )
        ## when derivation of function shall be returned
        if derivate:
            return elementwise_grad(y)
        return  y

    def center(self, X:np.array, verbose:int) -> np.array:
        '''
        function to center the signal by subtracting the mean
        Parameters:
            - X: array to center [numpy.array]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information
                - 1 -> more detailed information
        Returns:
            - X_new: centered array [numpy.array]
        '''
        ## calc mean
        mean = X.mean(axis=-1, keepdims=True)
        ## subtract mean from x values
        X_new = X - mean
        self.mean = mean
        if verbose > 0:
            print(f'Mean of data:\n{mean}')
        return X_new

    def whitening(self, X:np.array, n_components:int, n_features: int, verbose:int) -> np.array:
        '''
        function to whiten the signal by eigen-value decomposition of the covariance matrix
        Parameters:
            - X: array to whiten [numpy.array]
            - n_components: number of components to use [Integer]
            - n_features: number of features in the data [Integer]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information
                - 1 -> more detailed information
        Returns:
            - X_whiten: whitened array [numpy.array]
        '''
        ## make singular value decomposition
        u, d, _ = np.linalg.svd(X, full_matrices = False)
        del _
        ## calc diagonal matrix- reduce dimensions
        K = (u / d).T[:n_components]
        del u, d
        ## get whitened data
        X_whiten = np.dot(K, X)
        ## multiplicate with square root of the number of features
        X_whiten *= np.sqrt(n_features)
        if verbose > 0:
            print(f'Whitening matrix:\n{K}')
        return X_whiten, K

    def calculate_new_w(self, w:np.array, X:np.array, g:object, g_der:object, W:np.array, i:int) -> np.array:
        '''
        function to update the de-mixing matrix w
        Parameters:
            - w: de-mixing matrix w [numpy.array]
            - X: whitened array [numpy.array]
            - g: g function [object]
            - g_der: derivate of g function [object]
            - W: null space definition [numpy.array]
            - i: current index of components [Integer]
        Returns:
            - w_new: new calculated de-mixing matrix [numpy.array]
        '''
        ## calc new w
        w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
        ## rthonormalize w with respect to the first j rows of W
        w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
        ## rescale w
        w_new /= np.sqrt((w_new ** 2).sum())
        return w_new
                      
    def lda(self, X:np.array, verbose:int = 0) -> None:
        '''
        trains the lda projection for further predictions
        Parameters:
            - X: Train Data [numpy.array]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - None
        '''
        self.X = X
        self.mean = np.array([0 for _ in range(X.shape[0])])
        ## calculate the means of the classes
        means = np.array([self.calc_mean(x, verbose) for x in X])
        ## calculate the scatter matrix of the data
        S_W = self.calc_sw(means, verbose)
        ## calculate transforming vector
        self.components_ = np.linalg.inv(S_W).dot(means[0]-means[1]).T
        
    def pca(self, X:np.array, dim:int, verbose:int = 0) -> None:
        '''
        trains the pca for further transformation
        Parameters:
            - X: Train Data [numpy.array]
            - dim: desired dimension of projected data [Integer]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - None
        '''
        ## make sure desired dimension is smaller_equal than actual dimension of data or None
        if (dim == None) or (dim > X.shape[1]):
            print(f'setting dim to be {X.shape[1]}')
            dim = X.shape[1]
        ## calc mean
        mean = self.calc_mean(X, verbose)
        ## make data zero mean
        self.X = X - mean
        self.mean = mean
        ## calc covariance matrix
        cov = self.calc_cov(verbose)
        ## calc scatter matrix
        S_W = self.calc_sw(mean, verbose)
        ## calc (eigenvalue, eigenvector)-pairs
        eig_pairs = self.calc_eig_vals(S_W, cov, verbose)
        self.components_ = self.calc_W(eig_pairs, dim, verbose).T
        
    def fastica(self, X:np.array, n_components:int = None, iterations:int = 200, tolerance:float = 1e-4, verbose:int = 0) -> None:
        '''
        trains the ica for further predictions
        Parameters:
            - X: Train Data [numpy.array]
            - n_components: desired number of components of projected data [Integer, default = None]
            - iterations: max. number of iterations per dimension [Integer, default = 20]
            - tolerance: tolerance to break iteration [Float, default = 1e-5]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
        Returns:
            - None
        '''
        ## set random seed to have the same result every time
        np.random.seed(42)
        ## make sure X is numpy.array
        X = np.array(X).T
        ## get number of samples and number of features
        n_samples, n_features = X.shape
        ## make sure desired dimension is smaller_equal than actual dimension of data or None
        if (n_components == None) or (n_components > n_samples):
            print(f'setting n_components to be {n_samples}')
            n_components = n_samples
        ## center X
        X = self.center(X, verbose)
        ## whiten X
        X1, K = self.whitening(X, n_components, n_features, verbose)
        ## init de-mixing matrix of all components
        W = np.zeros((n_components, n_components), dtype = X.dtype)
        ## get g and g'
        g = self.f("exp")
        g_der = self.f("exp", True)
        ## go through all components that shall be kept
        for i in range(n_components):
            ## init w randomly, rescale it 
            w = np.random.rand(n_components)
            w /= np.sqrt((w ** 2).sum())
            ## start iterating
            for j in range(iterations):
                ## calc new w
                w_new = self.calculate_new_w(w, X1, g, g_der, W, i)
                ## w shall be almost orthogonal to w_new -> if so, then w x w_new == 1
                distance = np.abs(np.abs((w_new * w).sum()) - 1)
                ## reset w to be w_new
                w = w_new
                ## check if distance almost 1 -> break then
                if distance <= tolerance:
                    break
            ## add component of all components de-mixing matrix
            W[i,:] = w
        ## compute the components
        self.components_ = np.dot(W, K)
    
    def fit(self, X:np.array, algorithm:str, verbose:int = 0, dim:int = None, n_components:int = None, iterations:int = 200, tolerance:float = 1e-4) -> None:
        '''
        fits the model to X
        Parameters:
            - X: Train Data [numpy.array]
            - algorithm: which feature reduction algorithm to use. Possible values are [String]
                - LDA -> "lda"
                - PCA -> "pca"
                - FastICA -> "fastica"
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
            - dim: desired dimension of projected data [Integer] - if PCA is the algorithm
            - n_components: desired number of components of projected data [Integer, default = None] - if FastICA is the algorithm
            - iterations: max. number of iterations per dimension [Integer, default = 20] - if FastICA is the algorithm
            - tolerance: tolerance to break iteration [Float, default = 1e-5] - if FastICA is the algorithm
        Returns:
            - None
        '''
        ## make sure X is numpy.array
        X = np.array(X)
        if algorithm == "lda":
            self.lda(X, verbose)
        elif algorithm == "pca":
            self.pca(X, dim, verbose)
        elif algorithm == "fastica":
            self.fastica(X, n_components, iterations, tolerance, verbose)
        else:
            print("algorithm not implemented (yet)! Using LDA instead")
            self.lda(X, verbose)
        self.is_fitted = True
        
    def fit_transform(self, X:np.array, algorithm:str, verbose:int = 0, dim:int = None, n_components:int = None, iterations:int = 200, tolerance:float = 1e-4) -> np.array:
        '''
        fits the model to X and returns transformed X
        Parameters:
            - X: Train Data [numpy.array]
            - algorithm: which feature reduction algorithm to use. Possible values are [String]
                - LDA -> "lda"
                - PCA -> "pca"
                - FastICA -> "fastica"
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information
            - dim: desired dimension of projected data [Integer] - if PCA is the algorithm
            - n_components: desired number of components of projected data [Integer, default = None] - if FastICA is the algorithm
            - iterations: max. number of iterations per dimension [Integer, default = 20] - if FastICA is the algorithm
            - tolerance: tolerance to break iteration [Float, default = 1e-5] - if FastICA is the algorithm
        Returns:
            - X_new: transormed X data [numpy.array]
        '''
        self.fit(X, algorithm, verbose, dim, n_components, iterations, tolerance)
        X_new = self.transform(X)
        return X_new
    
    def transform(self, x_test:np.array) -> np.array:
        '''
        function to transform given data point
        Parameters:
            - x_test: point to transform and predict [numpy.array]
        Returns:
            - x_transformed: transformed data point [numpy.array]
        '''
        ## check whether model is fitted
        self.check_is_fitted()
        ## make sure x_test is numpy array, subtract mean
        x_test = np.array(x_test) - self.mean.T
        return np.dot(x_test, self.components_.T)