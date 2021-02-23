'''
class that implements the Expetation-Maximization algorithm using Gaussian Mixture Models
'''

## Imports
import numpy as np
from ..utils._helper import _classifier

class gmm(_classifier):
    
    def __init__(self):
        '''
        constructor of class - imports needed modules
        initializes:
            - log-likelihoods (ll_new, [Float])
            - priors (pis, [numpy.array])
            - means (mus, [numpy.array])
            - (co)variances (sigmas, [numpy.array])
        Returns:
            - None
        '''
        self._imports()
        self.ll = 0.0
        self.pis = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        
    def _imports(self) -> None:
        '''
        handles the imports for the class
        Parameters:
            - None
        Returns:
            - None
        '''
        global np, py, go
        import numpy, plotly
        np, py, go = numpy, plotly.offline, plotly.graph_objs
    
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
        x = np.linspace(xs.min(),xs.max(),int(variance/1e-1))
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
    
    def train(self, x:np.array, n:int = 2, tol:float = 1e-5, max_iter:int = 100) -> None:
        '''
        implementation of EM (Expectation-Maximization) Algorithm using Gaussian Mixture Models. Fits the Probability Density Functions on the given data
        Parameters:
            - x: data points the GMM is fitted on [numpy.array]
            - n: number of different pdfs [Integer, default = 2]
            - tol: tolerance when to stop the fitting [Float, default = 1e-5]
            - max_iter: maximal number of iterations before stopping the fit [Integer, default = 100]
        Returns:
            - None
        '''
        xs = super().train(x)
        ## set random seed
        np.random.seed(42)
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
            - y_pred: class of prediction [numpy.array]
        '''
        x_test = super().predict(x_test)
        x_test = x_test.reshape(-1,1)
        l = len(self.pis)
        ## calculate the responsibilities
        responsibilities = [[self.pis[k] * (self.gauss_pdf(x_test,self.mus[k],self.sigmas[k]) / sum([self.pis[k_]*self.gauss_pdf(x_test,self.mus[k_],self.sigmas[k_]) for k_ in range(l)]))] for k in range(l)]
        ## get the corresponding y_pred (=label) (the one with the highest responsibility)
        y_pred = np.argmax([np.sum(x) for x in responsibilities])
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