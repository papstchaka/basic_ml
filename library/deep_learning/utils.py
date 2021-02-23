'''
helper functions, only needed for deep learning purposes, especially the plotting process
'''

## Imports
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from ..utils._helper import convertSeconds

def plot_progress(metrics:dict, params:list, verbose:int) -> None:
    '''
    plots current loss curves and prints progress
    Parameters:
        - metrics: metrics to print and plot [Dictionary]
        - params: further parameters to use (current epoch and number of epochs) [List]
        - verbose: how detailed the train process shall be documented. Possible values are [Integer]
            - 0 -> no information (default)
            - 1 -> more detailed information
    Returns:
        - None
    '''
    (train_loss, train_metrics, test_loss, test_metrics) = metrics.values()
    (e, epochs, score, starttime, currenttime) = params
    epochtime = time.time() - currenttime
    clear_output(wait = True)
    length = 80 if verbose == 0 else 40
    progress = int(round((e * length - 1) / epochs, 0))
    if verbose > 0:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot()
        x = [i for i in range(train_loss.__len__())]
        ax.plot(x, train_loss, label="train-loss")
        ax.plot(x, test_loss, label="validation-loss")
        ax.set_title(f'Epoch {e+1}/{epochs}\n{"=" * (progress)}>{"_"*int(1.5*(length-progress))}\nTrain-Loss: {train_loss[-1]:.4f}; Train-{score.upper()}: {train_metrics[-1]:.4f}; Test-Loss: {test_loss[-1]:.4f}; Test-{score.upper()}: {test_metrics[-1]:.4f}\nEstimated time: {convertSeconds(currenttime-starttime)}<{convertSeconds(epochtime*(epochs-e))}, {(1/epochtime):.2f}it/s', loc="left")
        plt.legend()
        plt.show()
    else:
        print(f'Epoch {e+1}/{epochs}')
        print(f'{"=" * progress}>{"."*(length-progress-1)}')
        print(f'Train-Loss: {train_loss[-1]:.4f}; Train-{score.upper()}: {train_metrics[-1]:.4f}; Test-Loss: {test_loss[-1]:.4f}; Test-{score.upper()}: {test_metrics[-1]:.4f}')
        print(f'Estimated time: {convertSeconds(currenttime-starttime)}<{convertSeconds(epochtime*(epochs-e))}, {(1/epochtime):.2f}it/s')