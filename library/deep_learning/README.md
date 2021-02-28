# `deep_learning`

contains Deep Learning using Neural Networks

<br/><br/>

-------

## <a href="callbacks.py" target="_blank">`callbacks.py`</a>

implements different kinds of callbacks for deep learning.

Callbacks are used to control the `training` process of the neural network. Some of them are used to have a graphical purification of the process, others are used to prematurely stop the process or to adjust parameters like the `learning rate`. Some possible callbacks and a brief overview what a callback exactly is, can be found <a href="https://www.kdnuggets.com/2019/08/keras-callbacks-explained-three-minutes.html" target="_blank">`here`</a>. The following are implemented:

- `EarlyStopping` stops the `training` process as soon as the `training loss` or `validation loss` is not decreasing anymore for multiple epochs
- `StopOverfitting` preventing the model from overfitting by stopping the `training` process as soon as `training loss` or `validation loss` are starting to rise again

<br/><br/>

-------

## <a href="layers.py" target="_blank">`layers.py`</a>

implements different kinds of layers for a Neural Network.

A very good introduction about `Neural Networks` in common and the meaning of `Layers` as part of them, can be found <a href="https://wiki.pathmind.com/neural-network" target="_blank">`here`</a>. `Neural Networks` can be used for both, `Classification` and `Regression`. Their major advantages - compared to `Classic Machine Learning` methods is the fact that the user does not need to know anything (or just very little) about the structure of the data. Only minor `preprocessing` steps are need, such as a `Scaler` or `Normalizer`.

`Neural Networks` in common consist of multiple, different kinds of `Layers`. Each `Layer` performs a different kind of transformation of the given data. Multiple kinds of `Layers` exists. The following - most common - are implemented in this project.

- <a href="https://dev.to/sandeepbalachandran/machine-learning-dense-layer-2m4n" target="_blank">`Dense`</a> performing a simple mathematical transformation:

     __y__ = &sigma;(__W__ &bigotimes; __x__ + __b__) with
     
    | symbol                                        | meaning               |
    | --------------------------------------------- | --------------------- |
    | &sigma;(  )                                   | activation function   |
    | __x__ &in; &real;<sup>**n**</sup>             | input                 |
    | __W__ &in; &real;<sup>**m**&times;**n**</sup> | weights               |
    | __b__ &in; &real;<sup>**m**</sup>             | bias                  |
    | __y__ &in; &real;<sup>**m**</sup>             | output                |

The following `Layers` are described <a href="https://wiki.tum.de/display/lfdv/Layers+of+a+Convolutional+Neural+Network#LayersofaConvolutionalNeuralNetwork-FullyConnectedLayer" target="_blank">`here`</a>:

- `Convolution` performs a `filter-like` transformation, multiplying the input with a given (formally random initialized) `kernel`, in common of the shape **2**&times;**2** or **3**&times;**3** - can also be `3-dimensional` (**2**&times;**2**&times;**2**, or **3**&times;**3**&times;**3**)
- `Pooling` performing a `Dimensionality Reduction` of the output from the `Convolution` Layer: Different kinds of transformations, representing a - mostly **2**&times;**2** or **3**&times;**3** - `array` by one single value, are possible:
    
    - `Max-Pooling` uses the __max__(`array`)
    - `Average Pooling`: uses the __avg__(`array`)
- `ReLU` performing a simple transformation eliminating all negative values. Therefor the `Layer` changes all negative values into `0` with the formal function: __max__(`0`, `original_value`).

Furthermore added:

- <a href="https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-3-flattening" target="_blank">`Flatten`</a> flattens the output from given input shape (e.g. **64**&times;**4**) into a `1-dimensional` array (here **256**&times;**1**).
- <a href="https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/dropout_layer" target="_blank">`Dropout`</a> randomly drops several `nodes` inside the `Neural Network` structure in every epoch. Thereafter the network can get more robust the <a href="https://syncedreview.com/2020/09/13/a-closer-look-at-the-generalization-gap-in-large-batch-training-of-neural-networks/" target="_blank">`generalization gap`</a> can be closed.

<br/><br/>

-------

## <a href="networks.py" target="_blank">`networks.py`</a>

implements two kinds of Neural Networks for Deep Learning.

Both possible use cases for `Neural Networks` are implemented, a standard `Classifier` and a standard `Regressor`, both used for `Supervised Learning`. All needed steps for `training` and `prediction` - described <a href="https://towardsdatascience.com/how-do-we-train-neural-networks-edd985562b73" target="_blank">`here`</a> - are implemented:

- `check_scaled_data()` checks whether the data is properly scaled - `Neural Networks` can only work with data in the range of `0` to `1`. If not, a scaling is performed
- `forward()` does a single forward step through the given network
- `train_step()` performing a single training step
- `iterate_minibatches()` iterates over the data to get the minibatches for training
- `train()` does the actual training for given `train_x` and `train_y`
- `predict_step()` performing a single prediction step
- `predict()` does the actual prediction for given `test_x` and `test_y`

<br/><br/>

-------

## <a href="utils.py" target="_blank">`utils.py`</a>

helper functions, only needed for deep learning purposes, especially the plotting process. The following functions are implemented:

- `plot_progress()` gives an `up-to-date` view onto the `training` process, being able to graphically show the evolution of the `loss` (or any desired metric). It is working mostly similar to <a href="https://www.tensorflow.org/" target="_blank">`Tensorflow's`</a> <a href="https://www.tensorflow.org/tensorboard" target="_blank">`Tensorboard`</a>.