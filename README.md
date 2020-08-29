# BASIC ML

Repository holds a library of most common algorithms used manually without using the help of <a href="https://scikit-learn.org/stable/" target="_blank">`sklearn`</a> or other Machine Learning libraries.
Only used libraries are:
>   - built-in libraries that belong to Python
>   - <a href="https://numpy.org/" target="_blank">`numpy`</a>
>   - <a href="https://docs.scipy.org/doc/scipy/reference/index.html" target="_blank">`scipy`</a> - Only used for `dendrogram` function in `scipy.cluster.hierarchy`
>   - <a href="https://docs.python.org/3.0/library/heapq.html" target="_blank">`heapq`</a>
>   - <a href="https://plotly.com/" target="_blank">`plotly`</a>

which can be installed via
>       `pip install numpy scipy heapq plotly`

## Functionalities that are already implemented (state of 14.08.2020):
>   - Linear Regression (with single- and multi-dimensional data support) <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" target="_blank">`Linear Regression by sklearn`</a>
>   - Clustering (with more or less working dendrogram building function - still a few bugs in it) <a href="https://scikit-learn.org/stable/modules/clustering.html" target="_blank">`Clustering by sklearn`</a>
>   - Dimension Reduction Algorithms (LDA and PCA) <a href="https://scikit-learn.org/stable/modules/unsupervised_reduction.html" target="_blank">`Dimension Reduction by sklearn`</a>
>   - Gaussian Mixture Models with Expectation Maximization Algorithm <a href="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html" target="_blank">`GMM with EM by sklearn`</a>
>   - Gaussian Processes Regression <a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html" target="_blank">`GP by sklearn`</a>
>   - Reinforcement Learning <a href="https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/" target="_blank">`Reinforcement Learning Tutorial by PythonProgramming`</a>

## Usage:
In general all classes and functions can be used exactly as those which are implemented in <a href="https://scikit-learn.org/stable/" target="_blank">`sklearn`</a> with a `training` and a `predict` function. `score` functionality will be added later.

Algorithms that work exactly as describe above:
>   - Linear Regression
>   - Clustering
>   - Dimension Reduction
>   - Gaussian Mixture Models with Expectation Maximization Algorithm

Algorithms with different work-wise:
>   - Gaussian Processes --> training and prediction are inseperable from each other and because of that implemented in one function (named `train`). To use GPs, you have to provide `x_train`, `y_train` and `x_test` all to this function. It fits the algorithm and returns the regressed `y_test`.
>   - Reinforcement Learning --> since there is no prediction in the workwise of RL, there is no such function implemented. Furthermore there is no (or not yet) `train` function implemented, since the user is obliged to self-decide whether or not to use `Q-Learning` or `Action-Value-Iteration`.

## Supporting developers:
> -   <a href="https://github.com/papstchaka" target="_blank">`Alexander Christoph`</a>