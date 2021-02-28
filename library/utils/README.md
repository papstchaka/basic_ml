# `utils`

contains different kind of `helper` functions and classes

<br/><br/>

-------

## <a href="_helper.py" target="_blank">`_helper.py`</a>

implements abstract classes as support for the other algorithms - for example implementing the `score()` function for all classifiers and regressors.

Both possible use cases for `Machine Learning` algorithms are implemented, the abstract implementation of a standard `Classifier` and a standard `Regressor`, both used for `Supervised Learning`. Similar to <a href="https://scikit-learn.org/stable/" target="_blank">`sklearn`</a> the following functionalities are implemented

- `_classifier` - the standart `Classifier` with 
    - `train()` 
    - `predict()` 
    - `score()`
- `_regressor` - the standard `Regressor` with
    - `train()` 
    - `predict()` 
    - `score()`

Furthermore added functionality:

- `convertSeconds()` that converts seconds into hours, minutes and seconds

<br/><br/>

-------

## <a href="metrics.py" target="_blank">`metrics.py`</a>

implements several loss-/activation-functions (for deep_learning) and the metrics for `score()`-evaluation

The following functionality is implemented here:

- `get_activation_function()` providing different kinds of `activation functions` for the `Layers` of a `Neural Networks`. A short description about `activation functions` can be found <a href="https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/" target="_blank">`here`</a>. The following ones are implemented:
    - `Sigmoid`
    - `Tangens Hyperbolicus`
    - `Rectified Linear Unit`
    - `Leaky Rectified Linear Unit`
    - `Soft-Max`
- `loss_function()` provides different kind of `loss functions` used for the `training` of a `Neural Network`. A short introduction towards `loss functions` can be found <a href="https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de" target="_blank">`here`</a>. The following of the presented functions are implemented:
    - `Mean squared Error` for Regression
    - `Mean absolute Error` for Regression
    - `Mean squared logarithmic Error` for Regression
    - `Hinge` for binary classification
    - `Squared Hinge` for binary classification
    - `Cross Entropy` for binary classification - here called `Binary Cross Entropy`
    - `Categorical Cross Entropy` for multi-class classification - here called `Multi-Class Cross Entropy`
    - `Kullback-Leibler Divergence` for multi-class classification

    Furthermore added are:
    - <a href="https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions" target="_blank">`L1-norm Error`</a> for Regression. Same as `Mean absoulte Error` without normalizing
    - <a href="https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions" target="_blank">`L2-norm Error`</a> for Regression. Same as `Mean squared Error` without normalizing
    - <a href="https://en.wikipedia.org/wiki/Root-mean-square_deviation" target="_blank">`Root mean squared Error`</a> for Regression
    - <a href="https://phuctrt.medium.com/loss-functions-why-what-where-or-when-189815343d3f" target="_blank">`Huber`</a> for Regression
- `calc_rates()` calculates the true positives, the false positives, the true negatives and the false negatives each class &rightarrow; <a href="https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62" target="_blank">`What are these rates?`</a>
- `classifier_score()` calculates possible evaluation metrics - described <a href="https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9" target="_blank">`here`</a> - for a `Classifier`, including:
    - `Recall`
    - `Precision`
    - `Accuracy`
    - `F1 score`
    - <a href="https://statisticaloddsandends.wordpress.com/2020/01/23/what-is-balanced-accuracy/" target="_blank">`Balanced Accuracy`</a> being similar to `Accuracy` but paying respect to an unbalanced number of samples for different classes
- `regressor_score()` calculates possible evaluation metrics - described above - for a `Regressor`, including:
    - `L1-norm`
    - `L2-norm`
    - `Mean squared Error`
    - `Mean absolute Error`
    - `Root mean squared Error`

<br/><br/>

-------

## <a href="preprocessing.py" target="_blank">`preprocessing.py`</a>

implements classes and functions for data preprocessing:

- `MinMaxScaler` - Splits arrays or matrices into random train and test subsets, similar to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html" target="_blank">`MinMaxScaler`</a> by `sklearn`. <a href="https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02" target="_blank">`Here`</a> you will find a brief introduction into the meaning of `Scaling`, `Standardizing` and `Normalizing`
- `train_test_split` - implements a MinMaxScaler that scales all given data into the range from 0 to 1, similar to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">`train_test_split`</a> by `sklearn`. Why splitting your dataset into `train` and `test` samples is important, can be found <a href="https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/" target="_blank">`here`</a>.