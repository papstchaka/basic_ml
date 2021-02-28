# `reinforcement_learning`

contains Reinforcement Learning algorithms

<br/><br/>

-------

## <a href="action_value_iteration.py" target="_blank">`action_value_iteration.py`</a>

class that implements reinforcement learning using <a href="https://stat.ethz.ch/education/semesters/ss2016/seminar/files/slides/seminar_week6_DynamicProgramming.pdf" target="_blank">`synchronous action value iteration algorithm`</a>
The implemented algorithms can be trained to find a policy for finding the optimal way through the following two tasks

- `1-dimensional` with final states in the states `[1]` and `[10]`
  |   |    |    |    |
  | - | -- | -- | -- |
  | 1 | -1 | -1 | 10 |
  |   |    |    |    |

  &rightarrow; which should result in the following policy, `B` indicating a `blocked` state, `X` indicating a `final` state 

  |   |              |              |   |
  | - | ------------ | ------------ | - |
  | X | &rightarrow; | &rightarrow; | X |
  |   |              |              |   |

- `2-dimensional` with final states in the states `[1]` and `[-10]`
  |   |   |   |     |
  | - | - | - | --: |
  | 0 | 0 | 0 |   1 |
  | 0 | B | 0 | -10 |
  | 0 | 0 | 0 |   0 |
  |   |   |   |     | 
   
  &rightarrow; which should result in the following policy, `B` indicating a `blocked` state, `X` indicating a `final` state 

  |              |              |              |             |
  | ------------ | ------------ | ------------ | ----------- |
  | &rightarrow; | &rightarrow; | &rightarrow; | X           |
  | &uparrow;    | B            | &uparrow;    | X           |
  | &uparrow;    | &leftarrow;  | &uparrow;    | &leftarrow; |
  |              |              |              |             |

<br/><br/>

-------

## <a href="genetic_algorithm.py" target="_blank">`genetic_algorithm.py`</a>

implements the feature selection using a <a href="https://link.springer.com/article/10.1007/s11042-020-10139-6" target="_blank">`genetic algorithm`</a>. Goal is to reduce the number of features to use for the latter machine learning model to reduce processing time and capacity.

Genetic algorithms are mostly used to produce new - and thereafter more - data. Those samples can be either used to `train` a `Machine Learning` algorithm with more data or to `train` it with better data. `Genetic Algorithms` cannot be used for all tasks as not all kinds of data samples can take on random values (e.g. `Port-Numbers` in `network traffic` or `features` that were encoded - by <a href="https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/" target="_blank">`OneHotEncoding`</a>, etc.). The workwise of those algorithms are similar to `Darwin's` theory of evolution, simulating `mutations`, `crossovers` and `natural selection`.

<br/><br/>

-------

## <a href="hmm.py" target="_blank">`hmm.py`</a>

class that implements <a href="http://users.monash.edu/~lloyd/tildeMML/Structured/HMM/" target="_blank">`Hidden Markov Models (HMM's)`</a>, including:

- <a href="https://brilliant.org/wiki/markov-chains/" target="_blank">`Markov Processes`</a> to estimate the probability of a given sequence
- forward + backward procedure for probability estimation
- <a href="https://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf" target="_blank">`Viterbi-Algorithm`</a> for optimal state sequence estimation
- model-reestimation for calculating the optimal parameters

`HMM's` can be used to determine probabilities for different kind of sequences when the single subprobabilities of the sequence are only partially known.

<br/><br/>

-------

## <a href="q_learning.py" target="_blank">`q_learning.py`</a>

class that implements reinforcement learning using <a href="https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/" target="_blank">`Q Learning algorithm`</a>
The implemented algorithms can be trained to find a policy for finding the optimal way through the following two tasks

- `1-dimensional` with final states in the states `[1]` and `[10]`
  |   |    |    |    |
  | - | -- | -- | -- |
  | 1 | -1 | -1 | 10 |
  |   |    |    |    |

  &rightarrow; which should result in the following policy, `B` indicating a `blocked` state, `X` indicating a `final` state 

  |   |              |              |   |
  | - | ------------ | ------------ | - |
  | X | &rightarrow; | &rightarrow; | X |
  |   |              |              |   |

- `2-dimensional` with final states in the states `[1]` and `[-10]`
  |   |   |   |     |
  | - | - | - | --: |
  | 0 | 0 | 0 |   1 |
  | 0 | B | 0 | -10 |
  | 0 | 0 | 0 |   0 |
  |   |   |   |     | 
   
  &rightarrow; which should result in the following policy, `B` indicating a `blocked` state, `X` indicating a `final` state 

  |              |              |              |             |
  | ------------ | ------------ | ------------ | ----------- |
  | &rightarrow; | &rightarrow; | &rightarrow; | X           |
  | &uparrow;    | B            | &uparrow;    | X           |
  | &uparrow;    | &leftarrow;  | &uparrow;    | &leftarrow; |
  |              |              |              |             |