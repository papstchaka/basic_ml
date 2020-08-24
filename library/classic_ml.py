import numpy as np
from heapq import nsmallest, nlargest
import plotly.offline as py
import plotly.graph_objs as go
import itertools

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
    
    def __init__(self) -> None:
        '''
        constructor of class
        initializes:
            - w to be an empty array
            - dim (dimension) to be 0
        Returns:
            - None
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
            - border: Accuracy between w_(t) and w_(t-1), when to stop the iteration because of convergence [Float, default = 0.01]
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
    
class clustering:
    '''
    class that implements different approaches to do clustering:
        - k-means-clustering
        - non-uniform-binary-split
        - producing a dendrogram of trained cluster algorithm using different distances metrices:
            - average
            - single-linkage
            - full-linkage
    '''
    
    def __init__(self) -> None:
        '''
        constructor of class
        initializes:
            - labels to be an emtpy array
            - cluster_centers to be an emtpy array
        Returns:
            - None
        '''
        self.labels = np.array([])
        self.cluster_centers = np.array([])
    
    def k_means(self, x:np.array, cluster_centers:list = [], k:int = 3, epochs:float=np.infty, border:float = 1e-5) -> None:
        '''
        implements K-Means-Clustering
        Parameters:
            - x: X Data to train on [numpy.array]
            - cluster_centers: inital cluster centers [List, default = []]
            - k: number of custers to search for [Integer, default = 3]
            - epochs: max number of epochs to do [Float, default = np.infty]
            - border: Accuracy between loss_(t) and loss_(t-1), when to stop the iteration because of convergence [Float, default = 1e-5]
        Returns:
            - None
        '''
        ## make sure to have initial cluster center
        if len(cluster_centers) == 0:
            cluster_centers = [x[np.random.randint(len(x))] for _ in range(k)]
        ## init loss function
        old_loss = np.infty
        i = 0
        ## start iterating
        while True:
            ## init clusters and labels
            labels = []
            cluster = [[] for i in range(len(cluster_centers))]
            ## go through all points in X, check the corresponding distances and label the point to respective cluster (E-Step)
            for point in x:
                distances = [np.linalg.norm(point - cluster_centers[i]) for i in range(len(cluster_centers))]
                label = np.argmin(distances)
                labels.append(label)
                cluster[label].append(point)
            ## recalc loss
            loss = sum([sum([np.linalg.norm(point - cluster_centers[i])**2 for point in cluster[i]]) for i,_ in enumerate(cluster_centers)])
            ## recalc new cluster centers (M-Step)
            cluster_centers = [np.array(1/len(c)) * sum(c) if len(c)>0 else cluster_centers[i] for i,c in enumerate(cluster)]
            i += 1
            ## check for convergence or max epoch (to break)
            if i > epochs or np.abs(old_loss - loss) < border:
                break
            old_loss = loss
        self.labels = np.array(labels).astype(str)
        self.cluster_centers = np.array(cluster_centers)
        
    def non_uniform_binary_split(self, x:np.array, k:int = 3, v:list = []) -> None:
        '''
        implements Non-Uniform-Binary-Split-Clustering
        Parameters:
            - x: X Data to train on [numpy.array]
            - k: number of custers to search for [Integer, default = 3]
            - v: vector(s) that shall be used to change the cluster centers [List, default = []]
        Returns:
            - None
        '''
        ## calc initial cluster center and cluster
        cluster_centers = [np.array(1/len(x)) * sum(x)]
        cluster = [[] for i in range(len(cluster_centers)+1)]
        ## init v vectors if not already given
        start = np.array([np.random.randint(1,10)/100])
        v = v if len(v) == k-1 else [np.array([start for _ in range(x.shape[1])])/(i+1) for i in range(k-1)]
        ## start iterating
        while len(cluster_centers) < k:
            ## calc loss of each clusters
            loss = [sum([np.linalg.norm(point - cluster_centers[i])**2 for point in cluster[i]]) for i,_ in enumerate(cluster_centers)]
            ## init clusters and labels
            cluster = [[] for i in range(len(cluster_centers)+1)]
            labels = []
            ## remove class with biggest loss
            remove_class = cluster_centers[np.argmax(loss)]
            ## new cluster centers are same as old, excepting the cluster center of class with biggest loss
            cluster_centers = [el for el, skip in zip(cluster_centers, [np.allclose(c, remove_class) for c in cluster_centers]) if not skip]
            ## add new calculated cluster centers (of old class with biggest loss)
            cluster_centers.extend([remove_class+v[len(cluster_centers)],remove_class-v[len(cluster_centers)]])
            ## go through all points in X, check the corresponding distances and label the point to respective cluster (E-Step)
            for point in x:
                distances = [np.linalg.norm(point - cluster_centers[i]) for i in range(len(cluster_centers))]
                label = np.argmin(distances)
                labels.append(label)
                cluster[label].append(point)
            ## recalc new cluster centers (M-Step)
            cluster_centers = [np.array(1/len(c)) * sum(c) for c in cluster]
        self.labels = np.array(labels).astype(str)
        self.cluster_centers = np.array(cluster_centers)
        
    def matrix_helper(self, point:float, matched_clusters:np.array, mode:str="") -> float:
        '''
        helper function to determine the right distance triggered by desired mode
        Parameters:
            - point: Point to determine the distance to [Float]
            - matched_clusters: all points to determine the distance to [numpy.array]
            - mode: desired linkage type [String, default=""]
        Returns:
            - val: distance value [Float]
        '''
        if mode == "single-linkage":
            val = np.min([np.linalg.norm(point-punto) for punto in matched_clusters])
        elif mode == "full-linkage":
            val = np.max([np.linalg.norm(point-punto) for punto in matched_clusters])
        elif mode == "average":
            val = np.mean([np.linalg.norm(point-punto) for punto in matched_clusters])
        else:
            ## if the desired linkage type is not implemented
            print("this linkage does not exist")
            val = np.infty
        return val

    def make_distance_matrix(self, unmatched_clusters:np.array, matched_clusters:np.array=np.array([]), mode:str="") -> np.array:
        '''
        function to determine the distance matrix needed for the Z-Matrix
        Parameters:
            - unmatched_clusters: data of clusters which shall not be merged now [numpy.array]
            - matched_clusters: data of clusters which shall be merged now [numpy.array, default=numpy.array()]
            - mode: desired linkage type [String, default=""]
        Returns:
            - matrix: matrix with calculated distances [numpy.array]
        '''
        ## check whether a new cluster will be built or if all existing shall be analysed
        if len(matched_clusters)>0 and len(unmatched_clusters) >= len(matched_clusters):
            dim = len(unmatched_clusters) + 1
        else:
            dim = len(unmatched_clusters)
        ## init matrix
        matrix = np.zeros((dim, dim))
        ## if more clusters are not merged yet than merged ones (indicates it is not the last time this function will be called)
        if len(unmatched_clusters) >= len(matched_clusters):
            ## go through all data points
            for i,point in enumerate(unmatched_clusters):
                for j in range(i,len(unmatched_clusters)):
                    matrix[i][j] = np.linalg.norm(point - unmatched_clusters[j])
                    matrix[j][i] = np.linalg.norm(point - unmatched_clusters[j])   
                if len(matched_clusters)>0:
                    ## if a new cluster is created, fill it with corresponding (chosen by mode) values
                    newval = self.matrix_helper(point, matched_clusters, mode)
                    matrix[i][j+1], matrix[j+1][i] = newval, newval
        else:
            ## init minimum distance
            min_ = np.infty
            ## go through all data points
            for i,point in enumerate(unmatched_clusters):
                for j in range(len(unmatched_clusters)):
                    ## calculate new minimum distance, chosen by mode
                    nmin_ = self.matrix_helper(point, matched_clusters, mode)
                    ## if new one is smaller than old one, replace it
                    if nmin_ < min_:
                        min_ = nmin_
                    matrix[j][i], matrix[i][j] = min_, min_
                    ## all diagonal elements shall be 0
                    if i==j:
                        matrix[i][j], matrix[i][j] = 0,0
        ## all diagonal elements shall be infinity (that they are not considered as closest clusters)        
        matrix[matrix == 0] = np.infty
        return matrix

    def get_dendrogram_data(self, data:np.array, clusters:np.array, method:str = "single-linkage") -> np.array:
        '''
        calculates the data that is needed by the scipy.cluster.hierarchy.dendrogram function to plot the desired dendrogram
        Parameters:
            - data: data points for the dendrogram (usually the cluster centers) [numpy.array]
            - clusters: labels of the respective clusters [numpy.array]
            - method: desired linkage type [String, default="single-linkage"]
        Returns:
            - dd_data: data of dendrogram [numpy.array]
        NOTE:
        Function is still not working all the time. Sometimes some weird errors occur that are not fixed yet
        '''
        ## init dd_data
        dd_data = []
        ## init counter
        j=0
        ## get initial distance matrix
        matrix = self.make_distance_matrix(data,mode=method)
        ## make a copy of the original data
        initdata = data.copy()
        ## make sure cluster-labels are Integers
        clusters = np.unique(clusters).astype(int)
        ## make a copy of the original cluster-labels
        initclusters = clusters.copy()
        ## as long as not finished
        while len(data)>1:
            ## check the clusters with shortest distance
            matched_clusters = np.unique(np.where(matrix == np.min(matrix)),0)
            ## only two clusters at the time
            matched_clusters = matched_clusters[-2:]
            ## calc which clusters are not with shortest distance
            unmatched_clusters = np.setdiff1d(initclusters.astype(int),matched_clusters)
            ## make sure not to have higher index than length of data
            unmatched_clusters[unmatched_clusters >= len(data)-1] = len(data)-1
            ## drop all duplicates (of clusters in matched_clusters) in unmatched_cluster
            unmatched_clusters = np.setdiff1d(unmatched_clusters,matched_clusters)
            ## last iteration step
            if matrix.shape == (2,2):
                dd_data.append([clusters[-2],clusters[-1],np.min(matrix),clusters[-2]])
            else:
                dd_data.append([clusters[matched_clusters[0]],clusters[matched_clusters[1]],np.min(matrix),clusters[matched_clusters[0]]])
            ## if no unmatched_clusters remain --> end is reached
            if len(unmatched_clusters) == 0:
                unmatched_clusters = np.setdiff1d(initclusters,matched_clusters)
                unmatched_clusters, matched_clusters = matched_clusters, unmatched_clusters
            ## recalc distance matrix
            matrix = self.make_distance_matrix(data[unmatched_clusters],initdata[matched_clusters],mode=method)
            ## if not last iteration step reached
            if len(data)>2:
                data = data[unmatched_clusters]
            else:
                data = initdata[matched_clusters]
            ## rearange the clusters
            clusters = list(set(clusters.astype(int))-set(matched_clusters))
            clusters.append(len(initclusters)+j)
            clusters = np.array(sorted(clusters))
            j+=1

        dd_data = np.array(dd_data).astype(float)
        return dd_data

    def make_dendrogram(self, dd_data:np.array):
        '''
        uses sklearn.cluster.hierarchy.dendrogram function to draw a dendrogram from given data
        Parameters:
            - dd_data: Dendrogram data [numpy.array]
        Returns:
            - dendrogram: drawn Dendrogram [sklearn Plot]
        '''
        from scipy.cluster.hierarchy import dendrogram
        return dendrogram(dd_data)
    
    
    def train(self, x:np.array, mode:str = "k_means", k:int = 3) -> None:
        '''
        training function
        Parameters:
            - x: X Data to train on [numpy.array]
            - mode: desired mode - possible ones are [String]:
                - K-Means: --> "k_means" (default)
                - Non-Uniform-Binary-Split --> "nubs"
            - k: number of custers to search for [Integer, default = 3]
        Returns:
            - None
        '''
        ## be sure x is array
        x = np.array(x)
        if mode == "k_means":
            self.k_means(x, k=k)
        if mode == "nubs":
            self.non_uniform_binary_split(x, k=k)
    
    def predict(self, x_test:np.array) -> np.array:
        '''
        prediction function
        Parameters:
            - x_test: X Data to predict [numpy.array]
        Returns:
            - y_test: predicted Y Data [numpy.array]
        '''
        ## be sure x_test is array
        x_test = np.array(x_test)
        ## init y_test
        y_test = []
        ## get all unique found labels
        labels = list(set(self.labels))
        ## go through all points in x_test
        for point in x_test:
            ## calc distances to all cluster centers
            distances = [np.linalg.norm(point - c_center) for c_center in self.cluster_centers]
            ## add found label
            y_test.append(labels[np.argmin(distances)])
        ## be sure y_test is array
        y_test = np.array(y_test)
        return y_test
    
class dimension_reduction:
    '''
    class that implements different approaches of dimensionality reduction:
        - LDA: Linear Discriminant Analysis
        - PCA: Principal Component Analysis
    '''
    
    def __init__(self) -> None:
        '''
        constructor of class
        initializes:
            - W for transformation to 0
            - X as Train Data to emtpy numpy array
        Returns:
            - None
        '''
        self.W = 0
        self.X = np.array([])
        pass
    
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
        ## calculate the means of the classes
        means = np.array([self.calc_mean(x, verbose) for x in X])
        ## calculate the scatter matrix of the data
        S_W = self.calc_sw(means, verbose)
        ## calculate transforming vector
        self.W = np.linalg.inv(S_W).dot(means[0]-means[1])
        
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
        ## calc mean
        mean = self.calc_mean(X, verbose)
        ## make data zero mean
        self.X = X - mean
        ## calc covariance matrix
        cov = self.calc_cov(verbose)
        ## calc scatter matrix
        S_W = self.calc_sw(mean, verbose)
        ## calc (eigenvalue, eigenvector)-pairs
        eig_pairs = self.calc_eig_vals(S_W, cov, verbose)
        self.W = self.calc_W(eig_pairs, dim, verbose)
    
    def transform(self, x_test:np.array) -> np.array:
        '''
        function to transform given data point
        Parameters:
            - x_test: point to transform and predict [numpy.array]
        Returns:
            - transformed data point [numpy.array]
        '''
        return x_test.dot(self.W)
    
    def predict(self, x_test:np.array) -> np.array:
        '''
        transforms given data point and gives a class prediction using 1-nearest-classifier
        Parameters:
            - x_test: point to transform and predict [numpy.array]
        Returns:
            - class label (0,1,..,n) of transformed and classified data point [numpy.array]
        '''
        ## transform data point
        x_test = self.transform(x_test)
        ## calculate distances to all different classes
        dist = [min([ np.linalg.norm(x_test - c.dot(self.W)) for c in label ]) for label in self.X]    
        return np.argmin(dist)
    
class gmm:
    '''
    class that implements the Expetation-Maximization algorithm using Gaussian Mixture Models
    '''
    
    def __init__(self):
        '''
        constructor of class
        initializes:
            - log-likelihoods (ll_new, [Float])
            - priors (pis, [numpy.array])
            - means (mus, [numpy.array])
            - (co)variances (sigmas, [numpy.array])
        Returns:
            - None
        '''
        self.ll = 0.0
        self.pis = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
    
    def plot_pdf(self, xs:np.array, mus:np.array, sigmas:np.array, variance:int = 2) -> None:   
        '''
        Plots the given pdf with corresponding mean and the data points the GMM is fitted on
        Parameters:
            - xs: data points the GMM is fitted on [numpy.array]
            - mus: Means of the multiple pdfs [numpy.array]
            - sigmas: (Co)variances of mutliple pdfs [numpy.array]
            - variance: determines how many values shall be evaluated for the plot -> number of points = variance/0.1 [Integer, default = 2]
        Returns:
            - None
        '''
        ## set random seed
        np.random.seed(42)
        ## set number of different classes
        l = len(mus)
        ## init x values
        x = np.linspace(xs.min(),xs.max(),variance/1e-1)
        pos = np.array(x.flatten()).T.reshape(-1,1)   
        multidim = False
        ## if multivariate gaussian (more than 1D)
        if xs.shape[1] > 1:
            multidim = True
            X,Y = np.meshgrid(x,x)
            pos = np.array([X.flatten(),Y.flatten()]).T
            ## calc all the corresponding responsibilities
            Z = [np.array([self.gauss_pdf(l, mus[i], sigmas[i]).squeeze() for l in pos]).reshape(len(x),len(x)) for i in range(l)] 
        else:
            ## calc all the corresponding responsibilities
            Z = [np.array([self.gauss_pdf(l, mus[i], sigmas[i]).squeeze() for l in pos]).reshape(len(x)) for i in range(l)] 
        ## set color palette
        colors = [f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})' for color in np.random.randint(0,100,(l,3))/100]
        ## init plotting data
        data = []
        for i in range(l):
            ## if multivariate
            if multidim:
                ## add pdf
                data.append(go.Contour(x = x, y = x, z = Z[i],contours_coloring='lines',line_width=2, name=f'Class: {i}', line={'color':colors[i]}, showscale=False))
                ## add means
                data.append(go.Scatter(x = [mus[i][0]], y = [mus[i][1]], name=f'Mean of Class: {i}', marker_size=15, marker_color=colors[i], line={'color':colors[i]}, marker={"symbol":"x"}))
            else:
                ## add pdf
                data.append(go.Scatter(x = x , y = Z[i], name=f'Class: {i}', marker_color=colors[i], line={'color':colors[i]}))
                ## add means
                data.append(go.Scatter(x = [mus[i]], y=[0], name=f'Mean of Class: {i}', marker_size=15, marker_color=colors[i], line={'color':colors[i]}, marker={"symbol":"x"}))
        ## reshape data points
        x = xs[:,0].squeeze().tolist()
        y = [0 for _ in xs] if xs.shape[1] == 1 else xs[:,1].squeeze().tolist()
        ## add data points
        data.append(go.Scatter(x=x,y=y, mode="markers", marker_size=8,showlegend=False))
        fig = go.Figure(data)
        py.iplot(fig)
    
    def gauss_pdf(self, x:np.array, mu:np.array, sigma:np.array) -> float:
        '''
        function that implements a single- or multivariate gaussian probability density function
        Parameters:
            - x: Point to evaluate pdf of [numpy.array]
            - mu: Mean of pdf [numpy.array]
            - sigma: (Co)variance of pdf [numpy.array]
        Returns:
            - responsibility: Responsibility of given point x to belong to estimated pdf [Float]
        '''
        ## check whether 1D or multidimensional data
        if x.shape[0] == 1:
            ## check whether variance is 0 -> would return in pdf to be nan
            if sigma > 0:
                pdf = ( 1 / (np.sqrt(2 * np.pi) * sigma) ) * np.exp(-0.5 * ( (x-mu)/sigma )**2).squeeze()
                return pdf
        else:
            ## check whether variance is 0 -> would return in pdf to be nan
            if np.sum(sigma) > 0:
                pdf = ( 1 / (np.sqrt(2 * np.pi)**len(x) * np.linalg.det(sigma)) ) * np.exp(-0.5 * ( (x-mu).dot(np.linalg.inv(sigma)).dot((x-mu).T) ))
                return pdf
        return np.nan
    
    def train(self, xs:np.array, n:int = 2, tol:float = 1e-5, max_iter:int = 100) -> None:
        '''
        implementation of EM (Expectation-Maximization) Algorithm using Gaussian Mixture Models. Fits the Probability Density Functions on the given data
        Parameters:
            - xs: data points the GMM is fitted on [numpy.array]
            - n: number of different pdfs [Integer, default = 2]
            - tol: tolerance when to stop the fitting [Float, default = 1e-5]
            - max_iter: maximal number of iterations before stopping the fit [Integer, default = 100]
        Returns:
            - None
        '''
        ## set random seed
        np.random.seed(42)
        ## make sure xs is numpy.array
        xs = np.array(xs)
        ## check whether multivariate (> 1D) data
        if len(xs.shape) == 1:
            xs = xs.reshape(-1,1)
        ## sort x values
        xs.sort()
        ## init priors + log
        pis = [(1/n) for _ in range(n)]
        ## init dimensions
        n,p = xs.shape
        l = len(pis)
        ll_old = 0
        ## init means
        mus = []
        i = 0
        while i < l:
            ## init new random mean
            nmu = np.random.randint(xs.min(axis=0),xs.max(axis=0)).squeeze()
            ## if same mean wasn't already calculated
            if not any((nmu == x).all() for x in mus):
                mus.append(nmu)
                i+=1
        ## init (co)variances, scale them to be between 0 and 1
        sigmas = [ (1 / len(xs)) * sum( [(x-mu).reshape(1,-1) * (x-mu).T.reshape(-1,1) for x in xs] ) for mu in mus]
        sigmas = [ sigma / sigma.max() for sigma in sigmas]
        ## make sure priors, means and (co)variances are all numpy.arrays
        pis = np.array(pis)
        mus = np.array(mus)
        sigmas = np.array(sigmas)
        ## fit pdfs
        for _ in range(max_iter):
            ## plot data
            self.plot_pdf(xs, mus, sigmas)
            ll_new = 0
            ## E-Step - calculate responsibilities
            thetas = np.zeros((l,n))
            for k in range(l):
                for i in range(n):
                    theta = pis[k] * (self.gauss_pdf(xs[i],mus[k],sigmas[k]) / sum([pis[k_]*self.gauss_pdf(xs[i],mus[k_],sigmas[k_]) for k_ in range(l)]))
                    ## if responsibility is nan, set to zero
                    if np.isnan(np.sum(theta)):
                        theta = 0
                    thetas[k,i] = theta
            ## check whether one of the responsibilities equals 0. Then break imideatly
            if sum([ sum([1 if np.sum(x)==0 else 0 for x in theta]) for theta in thetas]) > 0:
                print("Can't proceed algorithm due to 'divide by zero occurences'")
                break
            ## rescale responsibilites - just in case that the sum of the priors is not 1
            thetas /= thetas.sum(0)
            ## M-Step - calculate new means, priors and (co)variances
            n_k = np.zeros((l))
            for k in range(l):
                n_k[k] = np.sum(thetas[k])
            mus = np.zeros((l,p))
            for k in range(l):
                mus[k] = ( 1 / n_k[k] ) * sum( [thetas[k][j] * (xs[j]) for j in range(n)] )
            pis = np.zeros((l))
            for k in range(l):
                pis[k] = n_k[k] / n
            sigmas = np.zeros((l,p,p))
            for k in range(l):
                sigmas[k] = (1 / n_k[k]) * sum( [thetas[k][j] * ((xs[j]-mus[k]).reshape(-1,1) * (xs[j]-mus[k])).reshape(-1,1) for j in range(n)] ).reshape(p,p)
            ## calculate new log-likelihood
            ll_new = np.sum([ np.log(np.sum([pis[k_]*self.gauss_pdf(xs[i],mus[k_],sigmas[k_]) for k_ in range(l)])) for i in range(n)])
            ## check whether changing tolerance were beaten
            if np.abs(ll_new - ll_old) < tol:
                break
            ll_old = ll_new
            ## drop all unnecessary dimensions
            pis, mus, sigmas = pis.squeeze(), mus.squeeze(), sigmas.squeeze()
        self.ll = ll_new
        self.pis = pis
        self.mus = mus
        self.sigmas = sigmas
        
    def predict(self, x_test:np.array) -> np.array:
        '''
        prediction function
        Parameters:
            - x_test: X Data to predict [numpy.array]
        Returns:
            - y_test: class of prediction [numpy.array]
        '''
        ## be sure x_test is array
        x_test = np.array(x_test).reshape(-1,1)
        l = len(self.pis)
        ## calculate the responsibilities
        responsibilities = [[self.pis[k] * (self.gauss_pdf(x_test,self.mus[k],self.sigmas[k]) / sum([self.pis[k_]*self.gauss_pdf(x_test,self.mus[k_],self.sigmas[k_]) for k_ in range(l)]))] for k in range(l)]
        ## get the corresponding label (the one with the highest responsibility)
        label = np.argmax([np.sum(x) for x in responsibilities])
        return label
    
class gp:
    '''
    class that implements gaussian processes as regression algorithm 
    '''
    
    def __init__(self):
        '''
        constructor of class
        initializes:
            - None
        Returns:
            - None
        '''
        pass
    
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
    
    def train(self, x:np.array, x_test:np.array, y:np.array, sigma:float, l:float, mode:str = "rbf", return_cov=False) -> tuple:
        '''
        training function for Gaussian Process given x and y data. Also predicts the regressed value(s) for given x_test
        Parameters:
            - x: training data points [numpy.array]
            - x_test: testing data points [numpy.array]
            - y: training labels [numpy.array]
            - sigma: noise constant to add [Float]
            - l: lenghtscale of GP [Float]
            - mode: which kind of kernel to use. Possible values are [String]
                - rbf: Radial Basis Function (default)
            - return_cov: whether (= True) or not (= False) return the covariance of GP [Boolean, default=False]
        Returns:
            if return_cov == True:
                - (y_test, covariance): tuple of regressed y_test and covariance [tuple]
            else:
                - y_test: regressed y_test [numpy.array]
        '''
        ## make sure x, x_test and y are numpy.arrays
        x, x_test, y = np.array(x), np.array(x_test), np.array(y)
        ## calc all kernel matrizes
        k, k_star, k_2star = self.calc_kernel(x,x_test,sigma,l,mode)
        ## get shape of x
        n = k.shape[0]
        ## get regressed y_test
        y_test = np.dot(k_star, np.dot(np.linalg.inv(k + (sigma)*np.eye(n)), (y.reshape([n, 1]))))
        ## calc covariance
        if return_cov:
            covariance = k_2star - np.dot(k_star, np.dot(np.linalg.inv(k + (sigma)*np.eye(n)), k_star.T))
            return (y_test, covariance)
        return y_test
