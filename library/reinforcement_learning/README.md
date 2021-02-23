# `reinforcement_learning`

contains Reinforcement Learning algorithms

<br/><br/>

-------

## <a href="action_value_iteration.py" target="_blank">`action_value_iteration.py`</a>

class that implements reinforcement learning using `synchronous action value iteration algorithm`
The implemented algorithms can be trained to find a policy for finding the optimal way through the following two tasks

    - [1, -1, -1, 10] which should lead to point in the direction of the 10 [1D]
    - [0, 0, 0, 1  ]
      [0, x, 0, -10] -> which should lead to point in the direction of the 1 [2D]
      [0, 0, 0, 0  ]

<br/><br/>

-------

## <a href="genetic_algorithm.py" target="_blank">`genetic_algorithm.py`</a>

implements the feature selection using a genetic algorithm. Goal is to reduce the number of features to use for the latter machine learning model to reduce processing time and capacity

<br/><br/>

-------

## <a href="hmm.py" target="_blank">`hmm.py`</a>

class that implements Hidden Markov Models, including:

    - Markov Processes to estimate the probability of a given sequence
    - forward + backward procedure for probability estimation
    - Viterbi-Algorithm for optimal state sequence estimation
    - model-reestimation for calculating the optimal parameters

<br/><br/>

-------

## <a href="q_learning.py" target="_blank">`q_learning.py`</a>

class that implements reinforcement learning using Q Learning algorithm
The implemented algorithms can be trained to find a policy for finding the optimal way through the following two tasks

    - [1, -1, -1, 10] which should lead to point in the direction of the 10 [1D]
    - [0, 0, 0, 1  ]
      [0, x, 0, -10] -> which should lead to point in the direction of the 1 [2D]
      [0, 0, 0, 0  ]