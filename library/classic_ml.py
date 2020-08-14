import numpy as np
from heapq import nsmallest, nlargest

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