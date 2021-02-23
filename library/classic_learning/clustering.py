'''
class that implements different approaches to do clustering:
    - k-means-clustering
    - non-uniform-binary-split
    - producing a dendrogram of trained cluster algorithm using different distances metrices:
        - average
        - single-linkage
        - full-linkage
'''

## Imports
import numpy as np
from ..utils._helper import _classifier

class clustering(_classifier):
    
    def __init__(self) -> None:
        '''
        constructor of class - imports needed modules
        initializes:
            - labels to be an emtpy array
            - cluster_centers to be an emtpy array
        Returns:
            - None
        '''
        self._imports()
        self.labels = np.array([])
        self.cluster_centers = np.array([])
        
    def _imports(self) -> None:
        '''
        handles the imports for the class
        Parameters:
            - None
        Returns:
            - None
        '''
        global np
        import numpy
        np = numpy
    
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
        
    def linkage(self, data:np.array, mode:str) -> np.array:
        '''
        helper function to determine the right distance triggered by desired mode
        Parameters:
            - data: data to determine the linkage distance from - two rows of the matrix [numpy.array]
            - mode: desired linkage type. Possible values are [String]
                - "single-linkage"
                - "full-linkage"
                - "average"
        Returns:
            - new_vals: array with the new values (depeding on the mode) [numpy.array]
        '''
        if mode == "single-linkage":
            new_vals = np.min(data, axis=0)
        elif mode == "full-linkage":
            new_vals = np.max(data, axis=0)
        elif mode == "average":
            new_vals = np.mean(data, axis=0)
        else:
            ## if the desired linkage type is not implemented
            print("this linkage does not exist! Single Linkage will be used")
            new_vals = np.min(data, axis=0)
        return new_vals

    def recalc_matrix(self, matrix:np.array, matched_clusters:np.array, mode:str) -> tuple:
        '''
        helper function to determine the right distance triggered by desired mode
        Parameters:
            - matrix: current distance matrix that shall be changed [numpy.array]
            - matched_clusters: all points that determine the new matrix [numpy.array]
            - mode: desired linkage type. Possible values are [String]
                - "single-linkage"
                - "full-linkage"
                - "average"
        Returns:
            - tuple containing two variables:
                - new_matrix: changed matrix depending on given matched_clusters and mode [numpy.array]
                - distance: distance between both closest children [float]
        '''
        ## make sure not to overwrite the original data
        new_matrix = matrix.copy()
        ## check the rows that contain the data of the 'merged' clusters
        new_rows = matrix[matched_clusters]
        ## define a helper variable (if data gets replaced, that not the same minimum will be taken multiple times)
        help_val = np.infty
        ## get the lowest distance between all the clusters
        distance = np.min(new_rows)
        ## get the new vals for the 'merged' clusters (depending on the linkage mode)
        new_vals = self.linkage(new_rows, mode)
        ## replace the minimum, the row and the column of the second cluster with the helper variable
        new_vals[new_vals==distance] = help_val
        new_matrix[matched_clusters[1]] = help_val
        new_matrix[:,matched_clusters[1]] = help_val
        ## replace the row and the columns of the first cluster with the new data
        new_matrix[matched_clusters[0]] = new_vals
        new_matrix[:,matched_clusters[0]] = new_vals
        return new_matrix, distance

    def get_counts(self, children:np.array, data:np.array) -> np.array:
        '''
        calculates the number of data points in each children cluster
        Parameters:
            - children: the pre-calculated children in the dendrogram tree from the cluster algorithm [numpy.array]
            - data: the clustered data [numpy.array]
        Returns:
            - counts: array with respective number of points per children cluster [numpy.array]
        NOTE!
            - function only works if the initial data are cluster centers or only contain one data point
        '''
        ## init counts
        counts = []
        ## go thorugh all children
        for i,child in enumerate(children):
            ## if last child is reached
            if i == len(children) - 1:
                ## append all data points, because all data points are in the 'overall' cluster
                counts.append(len(data))
            ## if both 'cluster names' are in the range of really existing clusters
            elif np.max(child) < len(data):
                ## then it is the 'merge' of two initial centers and the result contains two data points
                counts.append(2)
            ## if the 'clusters  name' doesn't exist in the list of labels
            else:
                ## then the resulting cluster contains the data points of itself (=1) and the ones from the cluster before (=counts[-1])
                counts.append(counts[-1]+1)
        return np.array(counts)

    def get_dendrogram_data(self, data:np.array, clusters:np.array, method:str = "single-linkage") -> np.array:
        '''
        calculates the data that is needed by the scipy.cluster.hierarchy.dendrogram function to plot the desired dendrogram
        Parameters:
            - data: data points for the dendrogram (usually the cluster centers) [numpy.array]
            - clusters: labels of the respective clusters [numpy.array]
            - method: desired linkage type. Possible values are [String]
                - "single-linkage" (default)
                - "full-linkage"
                - "average"
        Returns:
            - dd_data: data of dendrogram [numpy.array]
        '''
        ## make sure 'cluster names' are numeric and unique and a numpy.array
        clusters = np.array([i for i,_ in enumerate(np.unique(clusters))])
        ## init the matrix with all the distances (as zeros)
        matrix = np.zeros((len(data), len(data)))
        ## go through all data points
        for i, point in enumerate(data):
            for j, other in enumerate(data):
                ## except the diagonal elements - the are always 0
                if i!=j:
                    ## calc the distance between all the points and set to respective matrix element
                    matrix[i][j] = np.linalg.norm(point - other)
                ## replace diagonal elements with infitiy
                else:
                    matrix[i][j] = np.infty
                ## matrix is symmetric
                matrix[j][i] = matrix[i][j]
        ## get the biggest 'cluster name'
        max_label = np.max(clusters)
        ##init children and distances
        children = []
        distances = []
        ## as linkage matrix contains len(clusters)-1 rows and 4 columns (child1, child2, distance, count)
        for i in range(clusters.__len__() - 1):
            ## get the indize of the two clusters with the lowest distance in between
            matched_clusters = np.where( matrix == np.min(matrix) )
            matched_clusters = np.unique(matched_clusters,0)[-2:]
            ## recalculate the matrix
            matrix, new_distance = self.recalc_matrix(matrix, matched_clusters, method)
            ## check 'who' the new children are - sorted
            new_childs = sorted(clusters[matched_clusters])
            ## if children is still empty, start with initial fill of new children
            if children.__len__() == 0:
                children = [new_childs]
            ## else, append the new children
            else:
                children.append(new_childs)
            ## replace the 'cluster names' of the already merged ones with new ones (just counting up)
            clusters[matched_clusters] = max_label + i + 1
            ## add the current distance to the distances
            distances.append(new_distance)
        ## make sure children and distances are numpy.arrays
        children = np.array(children)
        distances = np.array(distances)
        ## get the counts
        counts = self.get_counts(children, data)
        ## set together the whole dendrogram data (being the scipy.cluster.hierarchy.linkage)
        ddata = np.column_stack([children, distances, counts]).astype(float)
        return ddata

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
        x = super().train(x)
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
            - y_pred: predicted Y Data [numpy.array]
        '''
        x_test = super().predict(x_test)
        ## init y_pred
        y_pred = []
        ## get all unique found labels
        labels = list(set(self.labels))
        ## go through all points in x_test
        for point in x_test:
            ## calc distances to all cluster centers
            distances = [np.linalg.norm(point - c_center) for c_center in self.cluster_centers]
            ## add found label
            y_pred.append(labels[np.argmin(distances)])
        ## be sure y_pred is array
        y_pred = np.array(y_pred)
        return y_pred
    
    def score(self, y_test:np.array, y_pred:np.array, mode:str = "accuracy") -> float:
        '''
        implements the scoring function for the classifiers, depending on given mode
        Parameters:
            - y_test: actual y-values - ground truth [numpy.array]
            - y_pred: predicted y-values - prediction [numpy.array]
            - mode: mode of the scoring function. Possible values are [String]
                - Recall --> "recall"
                - Precision --> "precision"
                - Accuracy --> "accuracy" = default
                - F1 --> "f1"
                - balanced Accuracy --> "balanced_accuracy"
            - average: whether (=True, default) or not (=False) to average the estimated score (=metric) over the given classes
        Returns:
            - score: calculated score [Float]
        '''
        return super().score(y_test, y_pred, mode)