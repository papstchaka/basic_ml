# BASIC ML

Repository holds a library of most common algorithms used manually without using the help of <a href="https://scikit-learn.org/stable/" target="_blank">`sklearn`</a> or other Machine Learning libraries.
Only used libraries are:
>   - built-in libraries that belong to Python
>   - <a href="https://numpy.org/" target="_blank">`numpy`</a>
>   - <a href="https://docs.scipy.org/doc/scipy/reference/index.html" target="_blank">`scipy`</a> - Only used for `dendrogram` function in `scipy.cluster.hierarchy`
>   - <a href="https://docs.python.org/3.0/library/heapq.html" target="_blank">`heapq`</a>
>   - <a href="https://plotly.com/" target="_blank">`plotly`</a>
>   - <a href="https://github.com/tqdm/tqdm" target="_blank">`tqdm`</a> - for showing progress of training
>   - <a href="https://github.com/HIPS/autograd" target="_blank">`autograd`</a> - for derivation of functions

which can be installed via
>       `pip install numpy scipy heapq plotly tqdm autograd`

## Scripts:
The implemented algorithms are splitted into three parts, given by the three different scripts in `library`:
>   - <a href="library/classic_ml.py" target="_blank">`classic_ml.py`</a> contains all "classic" Machine Learning algorithms excluding Reinforcement Learning and Deep Learning
>   - <a href="library/reinforcement_learning.py" target="_blank">`reinforcement_learning.py`</a> contains Reinforcement Learning algorithms
>   - <a href="library/deep_learning.py" target="_blank">`deep_learning.py`</a> contains Deep Learning using Neural Networks
>   - <a href="library/genetic_algorithm.py" target="_blank">`genetic_algorithm.py`</a> containing a genetic algorithm implementation for DataSet manipulation
>   - <a href="library/_helper_.py" target="_blank">`_helper_.py`</a> containing abstract classes as support for the other algorithms - for example implementing the `score()` functions

## Functionalities that are already implemented (state of 04.10.2020):
>   - Linear Regression (with single- and multi-dimensional data support) <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" target="_blank">`Linear Regression by sklearn`</a>
>   - Clustering <a href="https://scikit-learn.org/stable/modules/clustering.html" target="_blank">`Clustering by sklearn`</a>
>   - Dimension Reduction Algorithms (LDA and PCA) <a href="https://scikit-learn.org/stable/modules/unsupervised_reduction.html" target="_blank">`Dimension Reduction by sklearn`</a>
>   - Gaussian Mixture Models with Expectation Maximization Algorithm <a href="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html" target="_blank">`GMM with EM by sklearn`</a>
>   - Gaussian Processes Regression <a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html" target="_blank">`GP by sklearn`</a>
>   - Reinforcement Learning <a href="https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/" target="_blank">`Reinforcement Learning Tutorial by PythonProgramming`</a>
>   - Hidden Markov Models <a href="http://scikit-learn.sourceforge.net/stable/modules/hmm.html" target="_blank">`HMM by sklearn`</a>
>   - Deep Learning using Neural Networks <a href="https://www.tensorflow.org/tutorials/keras/classification" target="_blank">`Deep Learning by Tensorflow`</a>
>   - Genetic Algorithm for DataSet manipulation <a href=https://pypi.org/project/sklearn-genetic/ target="_blank">`Genetic algorithm by sklearn`</a>

## Usage:
In general all classes and functions can be used exactly as those which are implemented in <a href="https://scikit-learn.org/stable/" target="_blank">`sklearn`</a> with a `training()`, a `predict()` and a `score()` - if possible - function.

Algorithms that work exactly as describe above:
>   - Linear Regression -> Regressor
>   - Clustering -> Classifier
>   - Dimension Reduction - does not support `score()` function
>   - Gaussian Mixture Models with Expectation Maximization Algorithm -> Classifier
>   - Deep Learning using Neural Networks - has its one `score()` function with the `loss()` function in training

Algorithms with different work-wise:
>   - Gaussian Processes --> training and prediction are inseperable from each other and because of that implemented in one function (named `train()`). To use GPs, you have to provide `x_train`, `y_train` and `x_test` all to this function. It fits the algorithm and returns the regressed `y_test`. Has a `score()` function
>   - Reinforcement Learning --> since there is no prediction in the workwise of RL, there is no such function implemented. Furthermore there is no (or not yet) `train()` function implemented, since the user is obliged to self-decide whether or not to use `Q-Learning` or `Action-Value-Iteration`.
>   - Hidden Markov Models --> as they need a sequence to be trained and initial states and observations, the class is used slightly different to `sklearn`-typical work wise. You have to provide a sequence to all of the implemented algorithms, further instructions can be found on top of the class description in the <a href="library/reinforcement_learning.py" target="_blank">`reinforcement_learning.py`</a> script.
>   - Genetic Algorithm for DataSet manipulation --> since there is no prediction in the workwise of GA, there is no such function implemented. The `train()` function gives back the best 'subdataset' that exists (in the original dataset or mutated from that)

## What's next:

>   - go through all the algorithms and add further variety

## Supporting developers:
> -   <a href="https://github.com/papstchaka" target="_blank">`Alexander Christoph`</a>