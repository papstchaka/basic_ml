'''
class that implements Hidden Markov Models, including:
    - Markov Processes to estimate the probability of a given sequence
    - forward + backward procedure for probability estimation
    - Viterbi-Algorithm for optimal state sequence estimation
    - model-reestimation for calculating the optimal parameters
'''

## Imports
import numpy as np

class hmm:
    
    def __init__(self, A:np.array, B:np.array, pi:dict, states:dict, observations:dict):
        '''
        constructor of class
        initializes:
            - A: matrix of state transition probability [numpy.array]
            - B: matrix of observations probability [numpy.array]
            - pi: dictionary of initial state probability [Dictionary] -> possbile states as string keys, corresponding probabilities as values
            - states: dictionary of possible states [Dictionary] -> possible states as string keys, 'index' as values
            - observations: dictionary of possible observatons [Dictionary] -> possible observations as string keys, 'index' as values
        Returns:
            - None
        '''
        ## be sure that sum of all probabilities is equal to 1 for A, B and pi
        A = (A.T/A.sum(1)).T
        B = (B.T/B.sum(1)).T
        for i,key in enumerate(pi.keys()):
            pi.update({key: list(list(pi.values()) / np.sum(list(pi.values())))[i]})        
        ## init all the model paramaters given
        self.A = A
        self.B = B
        self.pi = pi
        self.states = states
        self.observations = observations
        
    def check_sequence(self, sequence:list) -> int:
        '''
        checks whether sequence is 1D (which means only states are given as sequence) or 2D (means states and observations are given as sequence)
        Parameters:
            - sequence: list of given states or (observations, states) [List]
        Returns:
            - dimension: 1 if 1D, 2 if 2D [Integer]
        '''
        ## check if tuple in list --> indicates 2D problem
        if any((type(x) == tuple) for x in sequence):
            return 2
        return 1

    def dim1(self, sequence:list) -> float:
        '''
        implementation of 1D Markov Process
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - probability: probability of given sequence [Float]
        '''
        ## get first state
        init_state = sequence[0]
        ## init probability -> probability of model to be in first state
        probability = self.pi[init_state]
        ## go through rest of the sequence
        for i,state in enumerate(sequence[1:]):
            ## multiply with respective sub probabilities
            probability *= self.A[self.states[init_state],self.states[state]]
            ## reset 'first state' 
            init_state = state
        return probability

    def dim2(self, sequence:list) -> float:
        '''
        implementation of 2D Markov Process
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - probability: probability of given sequence [Float]
        '''
        ## init probability
        probability = 1
        ## go through the sequence
        for i, (s0,s1) in enumerate(sequence):
            ## multiply with respective sub probabilities
            probability *= self.B[self.states[s1], self.observations[s0]]
        return probability

    def MP(self, sequence:list) -> float:
        '''
        implementation of Markov Process. Can handle 1D (which means only states are given as sequence) or 2D (means states and observations are given as sequence) sequences
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - prob: probability of given sequence [Float]
        '''
        ## calc the dimension of the sequence
        dim = self.check_sequence(sequence)
        if dim == 1:
            prob = self.dim1(sequence)
        elif dim == 2:
            prob = self.dim2(sequence)
        ## throw exception because sequence can't be processed
        else:
            raise Exception("probability can't be calculated")
        return prob
    
    def forward(self, sequence:list) -> float:
        '''
        implementation of forward procedure for probability estimation
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - prob: probability of given sequence [Float]
        '''
        ## init forward process
        self.fwd = [{}]     
        ## initialize base cases (t == 0)
        for y in self.states.keys():
            self.fwd[0][y] = self.pi[y] * self.B[self.states[y]][self.observations[sequence[0]]]
        ## run forward algorithm for t > 0
        for t in range(1, len(sequence)):
            self.fwd.append({})     
            for y in self.states.keys():
                self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[self.states[y0],self.states[y]] * self.B[self.states[y],self.observations[sequence[t]]]) for y0 in self.states)
        prob = sum((self.fwd[len(sequence) - 1][s]) for s in self.states)
        return prob
    
    def backward(self, sequence:list) -> float:
        '''
        implementation of backward procedure for probability estimation
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - prob: probability of given sequence [Float]
        '''
        ## init backward process, get length of sequence
        self.bwk = [{} for t in range(len(sequence))]
        T = len(sequence)
        ## initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #A[y,"Final"] #pi[y] * B[y,observations[sequence[0]]]
        ## run backward algorithm for t < T
        for t in reversed(range(T-1)):
            for y in self.states:
                self.bwk[t][y] = sum((self.bwk[t+1][y1] * self.A[self.states[y],self.states[y1]] * self.B[self.states[y1],self.observations[sequence[t+1]]]) for y1 in self.states)
        prob = sum((self.pi[y]* self.B[self.states[y],self.observations[sequence[0]]] * self.bwk[0][y]) for y in self.states)
        return prob
    
    def viterbi(self, sequence:list) -> tuple:
        '''
        implementation of Viterbi-Algorithm for optimal state sequence estimation
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - tuple of [Tuple]
                - prob: probability of given sequence [Float]
                - path: estimated optimal state sequence [Lists]
        '''
        ## init path and viterbi process
        vit = [{}]
        path = {}     
        ## initialize base cases (t == 0)
        for y in self.states:
            ## add probability
            vit[0][y] = self.pi[y] * self.B[self.states[y],self.observations[sequence[0]]]
            ## add path
            path[y] = [y]
        ## run Viterbi for t > 0
        for t in range(1, len(sequence)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                ## get 'best' next state --> best subpath
                (prob, state) = max((vit[t-1][y0] * self.A[self.states[y0],self.states[y]] * self.B[self.states[y],self.observations[sequence[t]]], y0) for y0 in self.states)
                ## add probability
                vit[t][y] = prob
                ## add path
                newpath[y] = path[state] + [y]     
            ## don't need to remember the old paths
            path = newpath
        ## if only one element is observed max is sought in the initialization values
        n = 0           
        if len(sequence)!=1:
            n = t
        ## get 'best' overall path
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])
    
    def model_estimation(self, sequence:list) -> None:
        '''
        implements model-reestimation for calculating the optimal parameters
        Parameters:
            - sequence: desired sequence [List]
        Returns:
            - None
        '''
        ## this is needed to keep track of finding a state i at a time t for all i and all t
        gamma = [{} for t in range(len(sequence))]
        ## this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        zi = [{} for t in range(len(sequence) - 1)]  
        ## get alpha and beta tables computes (from forward and backward processes)
        p_obs = self.forward(sequence)
        _ = self.backward(sequence)
        ## compute gamma values
        for t in range(len(sequence)):
            for y in self.states:
                ## compute the new gammas
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    ## set new pi
                    self.pi[y] = gamma[t][y]
                ## compute zi values up to T - 1
                if t == len(sequence) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[self.states[y],self.states[y1]] * self.B[self.states[y1],self.observations[sequence[t + 1]]] * self.bwk[t + 1][y1] / p_obs
        ## re-estimate after getting new gamma and zi
        for y in self.states:
            for y1 in self.states:
                ## compute new a_ij
                val = sum([zi[t][y][y1] for t in range(len(sequence) - 1)]) #
                val /= sum([gamma[t][y] for t in range(len(sequence) - 1)])
                self.A[self.states[y],self.states[y1]] = val
        for y in self.states:
            ## for all possible observations
            for k in self.observations:
                ## compute new B
                val = 0.0
                for t in range(len(sequence)):
                    if sequence[t] == k :
                        val += gamma[t][y]                 
                val /= sum([gamma[t][y] for t in range(len(sequence))])
                self.B[self.states[y],self.observations[k]] = val