# `utils`

contains different kind of `helper` functions and classes

<br/><br/>

-------

## <a href="_helper.py" target="_blank">`_helper.py`</a>

implements abstract classes as support for the other algorithms - for example implementing the `score()` function for all classifiers and regressors

<br/><br/>

-------

## <a href="metrics.py" target="_blank">`metrics.py`</a>

implements several loss-/activation-functions (for deep_learning) and the metrics for `score()`-evaluation

<br/><br/>

-------

## <a href="preprocessing.py" target="_blank">`preprocessing.py`</a>

implements classes and functions for data preprocessing:

>    - `MinMaxScaler` - Splits arrays or matrices into random train and test subsets, similar to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html" target="_blank">`MinMaxScaler`</a> by `sklearn`
>    - `train_test_split` - implements a MinMaxScaler that scales all given data into the range from 0 to 1, similar to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">`train_test_split`</a> by `sklearn`