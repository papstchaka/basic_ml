import numpy as np

'''
implements Reinforcement Learning algorithms.
'''

class reinforcement_learning:
    '''
    class that implements reinforcement learning
    The implemented algorithms can be trained to find a policy for finding the optimal way through the following two tasks
        - [1, -1, -1, 10] which should lead to point in the direction of the 10 [1D]
        - [0, 0, 0, 1  ]
          [0, x, 0, -10] -> which should lead to point in the direction of the 1 [2D]
          [0, 0, 0, 0  ]
    '''
    
    def __init__(self):
        '''
        constructor of class
        initializes:
            - None
        Returns:
            - None
        '''
        pass
    
    def action_value_1D(self, states:np.array, actions:np.array, rewards:np.array, terminal_states:np.array, probabilities:np.array, gamma:float, tolerance:float, verbose:int) -> np.array:
        '''
        implements the 1D version of action value iteration algorithm
        Parameters:
            - states: all different possible states [numpy.array]
            - actions: all possible actions to do [numpy.array]
            - rewards: all different rewards for the different states - same shape as 'states'! [numpy.array]
            - terminal_states: all states which indicate 'the end of the search' [numpy.array]
            - probabilities: probabilities to do the desired - respective one of the other - action [numpy.array]
            - gamma: rate of forgetfulness - usually something like 0.9 [Float]
            - tolerance: tolerance to break the algorithm (convergence criteria) - usually something like 0.1 [Float]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information like Q(s,a) and V(s)
        Returns:
            - pi: trained policy [numpy.array]
        '''
        ## init a new and an old Q(s,a)
        q_k = np.zeros((len(states), len(actions)))
        new_q_k = q_k.copy()
        while True:
            ## go through all states and actions
            for i, state in enumerate(states):
                for j, action in enumerate(actions):
                    ## if current state is an end state
                    if state in terminal_states:
                        break
                    ## value for 'desired' action
                    val = probabilities[0] * (rewards[i+action] + gamma*np.max( q_k[i+action] ))
                    others = [i for i in range(len(probabilities)) if i!=j]
                    for l,other in enumerate(others):
                        ## value for all 'undesired' actions
                        val += probabilities[l+1] * (rewards[i+actions[other]] + gamma*np.max( q_k[i+actions[other]] ))
                    new_q_k[i,j] = round(val,2)
            ## stop when convergence reached
            if np.sum(np.abs(new_q_k - q_k)) < tolerance:
                break
            q_k = new_q_k.copy()
        v_k = [np.max(row) if i+1 not in terminal_states else rewards[i] for i,row in enumerate(q_k)]
        pi_k = [actions[np.argmax(row)] if i+1 not in terminal_states else "X" for i,row in enumerate(q_k)]    
        if verbose > 0:
            print(f'Determined Q(s,a):\n {new_q_k}')
            print(f'Determined V(s):\n {v_k}')
        return pi_k
    
    def action_value_2D(self, states:np.array, actions:np.array, rewards:np.array, terminal_states:np.array, probabilities:np.array, gamma:float, tolerance:float, verbose:int) -> np.array:
        '''
        implements the 2D version of action value iteration algorithm
        Parameters:
            - states: all different possible states [numpy.array]
            - actions: all possible actions to do [numpy.array]
            - rewards: all different rewards for the different states - same shape as 'states'! [numpy.array]
            - terminal_states: all states which indicate 'the end of the search' [numpy.array]
            - probabilities: probabilities to do the desired - respective one of the other - action [numpy.array]
            - gamma: rate of forgetfulness - usually something like 0.9 [Float]
            - tolerance: tolerance to break the algorithm (convergence criteria) - usually something like 0.1 [Float]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information like Q(s,a) and V(s)
        Returns:
            - pi: trained policy [numpy.array]
        '''
        ## two lambda functions to determine 'the next state'
        ac = lambda x,i: (x[0],x[1]+i)
        ac0 = lambda x: (x[0]+1,x[1])
        ## init a new and an old Q(s,a)
        q_k = np.zeros((states.shape[0], states.shape[1], actions.shape[0]))
        new_q_k = q_k.copy()
        while True:
            ## go through all states and actions
            for state, i in np.ndenumerate(states):
                for j, action in enumerate(actions):
                    ## if current state is an end state
                    if any(((state) == x).all() for x in terminal_states) or states[state] == "x":
                        break
                    ## determine next state
                    nstate = ac0(state) if action == 0 else ac(state,action)
                    ## check that 'next state' still a valid state - else 'bounce back' to current state
                    if (np.array(nstate)<(0,0)).any() or (np.array(nstate)>=rewards.shape).any() or states[nstate] == "x":
                        nstate = state            
                    val = probabilities[0] * (rewards[nstate] + gamma*np.max( q_k[nstate] ))
                    ## value for 'desired' action
                    others = [i for i in range(len(probabilities)) if i!=j]
                    for l,other in enumerate(others):
                        nstate = ac0(state) if actions[other] == 0 else ac(state,actions[other])
                        ## check that 'next state' still a valid state - else 'bounce back' to current state
                        if (np.array(nstate)<(0,0)).any() or (np.array(nstate)>=rewards.shape).any() or states[nstate] == "x":
                            nstate = state
                        ## value for all 'undesired' actions
                        val += probabilities[l+1] * (rewards[nstate] + gamma*np.max( q_k[nstate] ))
                    new_q_k[state[0],state[1],j] = round(val,2)
            ## stop when convergence reached
            if np.sum(np.abs(new_q_k - q_k)) < tolerance:
                break
            q_k = new_q_k.copy()
        v_k = np.array([[np.max(val) if (states[i,j] != "x") and not (any(((i,j) == x).all() for x in terminal_states)) else rewards[i,j] for j,val in enumerate(row)] for i,row in enumerate(q_k)])
        pi_k = np.array([[actions[np.argmax(val)] if (states[i,j] != "x") and not (any(((i,j) == x).all() for x in terminal_states)) else "B" if not (any(((i,j) == x).all() for x in terminal_states)) else "X" for j,val in enumerate(row)] for i,row in enumerate(q_k)])
        if verbose > 0:
            print(f'Determined Q(s,a):\n {new_q_k[::-1]}')
            print(f'Determined V(s):\n {v_k[::-1]}')
        return pi_k[::-1]

    def action_value_iteration(self, states:np.array, actions:np.array, rewards:np.array, terminal_states:np.array, probabilities:np.array, gamma:float, tolerance:float, verbose:int = 0) -> np.array:
        '''
        implements a 1D and a 2D version of action value iteration algorithm
        Parameters:
            - states: all different possible states [numpy.array]
            - actions: all possible actions to do [numpy.array]
            - rewards: all different rewards for the different states - same shape as 'states'! [numpy.array]
            - terminal_states: all states which indicate 'the end of the search' [numpy.array]
            - probabilities: probabilities to do the desired - respective one of the other - action [numpy.array]
            - gamma: rate of forgetfulness - usually something like 0.9 [Float]
            - tolerance: tolerance to break the algorithm (convergence criteria) - usually something like 0.1 [Float]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information like Q(s,a) and V(s)
        Returns:
            - pi: trained policy [numpy.array]
        '''
        ## Formula for Action-Value-Iteration: q_k+1(s,a) = sum( p(s',r | s, a) * [r + y * max( q_k(s',a') )] )
        ## check the shape of the states to determine whether 1D or 2D problem
        if len(states.shape) == 1:
            pi = self.action_value_1D(states, actions, rewards, terminal_states, probabilities, gamma, tolerance, verbose)
        elif len(states.shape) == 2:
            pi = self.action_value_2D(states, actions, rewards, terminal_states, probabilities, gamma, tolerance, verbose)
        return pi
    
    def q_learning_1D(self, states:np.array, actions:np.array, rewards:np.array, terminal_states:np.array, sequence:np.array, transition:dict, gamma:float, alpha:float, verbose:int) -> np.array:
        '''
        implements a 1D and a 2D version of action value iteration algorithm
        Parameters:
            - states: all different possible states [numpy.array]
            - actions: all possible actions to do [numpy.array]
            - rewards: all different rewards for the different states - same shape as 'states'! [numpy.array]
            - terminal_states: all states which indicate 'the end of the search' [numpy.array]
            - probabilities: probabilities to do the desired - respective one of the other - action [numpy.array]
            - sequence: sequence of actions that shall be done for training [numpy.array]
            - transition: dictionary that translates the sequence into the action [Dictionary]
            - gamma: rate of forgetfulness - usually something like 0.9 [Float]
            - alpha: learning rate - usually something like (1/len(actions)) [Float]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information like Q(s,a) and V(s)
        Returns:
            - pi: trained policy [numpy.array]
        '''
        ## init Q(s,a) and set initial state and starting state
        q_k = np.zeros((len(states), len(actions)))
        init_state = 1
        state = init_state
        ## go through whole sequence
        for move in sequence:
            ## translate action
            action = transition[move]
            ## translate 'right' = 1 to 1 and 'left' = -1 to 0 (index problem else)
            i = action if action>0 else 0
            q_k[state,i] = q_k[state,i] + alpha * (rewards[state+action] + gamma * np.max( q_k[state+action] ) - q_k[state,i])
            ## set next state
            state += action
            ## if next state is an end state, reset next state to be start state
            if state in terminal_states:
                state = init_state
        v_k = [np.max(row) if i not in terminal_states else rewards[i] for i,row in enumerate(q_k)]
        pi_k = [actions[np.argmax(row)] if i not in terminal_states else "X" for i,row in enumerate(q_k)]
        if verbose > 0:
            print(f'Determined Q(s,a):\n {q_k}')
            print(f'Determined V(s):\n {v_k}')
        return pi_k
    
    def q_learning_2D(self, states:np.array, actions:np.array, rewards:np.array, terminal_states:np.array, sequence:np.array, transition:dict, gamma:float, alpha:float, verbose:int) -> np.array:
        '''
        implements a 1D and a 2D version of action value iteration algorithm
        Parameters:
            - states: all different possible states [numpy.array]
            - actions: all possible actions to do [numpy.array]
            - rewards: all different rewards for the different states - same shape as 'states'! [numpy.array]
            - terminal_states: all states which indicate 'the end of the search' [numpy.array]
            - probabilities: probabilities to do the desired - respective one of the other - action [numpy.array]
            - sequence: sequence of actions that shall be done for training [numpy.array]
            - transition: dictionary that translates the sequence into the action [Dictionary]
            - gamma: rate of forgetfulness - usually something like 0.9 [Float]
            - alpha: learning rate - usually something like (1/len(actions)) [Float]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information like Q(s,a) and V(s)
        Returns:
            - pi: trained policy [numpy.array]
        '''
        ## two lambda functions to determine 'the next state'
        ac = lambda x,i: (x[0],x[1]+i)
        ac0 = lambda x: (x[0]+1,x[1])
        ## init Q(s,a) and set initial state and starting state
        q_k = np.zeros((states.shape[0], states.shape[1], actions.shape[0]))
        init_state = (0,0)
        state = init_state
        ## go through whole sequence
        for move in sequence:
            ## translate action
            action = transition[move]
            ## determine next state
            nstate = ac0(state) if action == 0 else ac(state,action)
            ## check that 'next state' still a valid state - else 'bounce back' to current state
            if (np.array(nstate)<(0,0)).any() or (np.array(nstate)>=rewards.shape).any() or states[nstate] == "x":
                nstate = state            
            ## translate 'right' = 1 to 2, 'up' = 0 to 1 and 'left' = -1 to 0 (index problem else)            
            i = action+1
            q_k[state[0],state[1],i] = q_k[state[0],state[1],i] + alpha * (rewards[nstate]
                                                                + gamma * np.max( q_k[nstate] ) - q_k[state[0],state[1],i])
            ## set next state
            state = nstate
            ## if next state is an end state, reset next state to be start state
            if any(((state) == x).all() for x in terminal_states):
                state = init_state
        v_k = np.array([[np.max(val) if (states[i,j] != "x") and not (any(((i,j) == x).all() for x in terminal_states)) else rewards[i,j] for j,val in enumerate(row)] for i,row in enumerate(q_k)])
        pi_k = np.array([[actions[np.argmax(val)] if (states[i,j] != "x") and not (any(((i,j) == x).all() for x in terminal_states)) else "B" if not (any(((i,j) == x).all() for x in terminal_states)) else "X" for j,val in enumerate(row)] for i,row in enumerate(q_k)])
        if verbose > 0:
            print(f'Determined Q(s,a):\n {q_k[::-1]}')
            print(f'Determined V(s):\n {v_k[::-1]}')
        return pi_k[::-1]
    
    def q_learning(self, states:np.array, actions:np.array, rewards:np.array, terminal_states:np.array, sequence:np.array, transition:dict, gamma:float, alpha:float, verbose:int = 0) -> np.array:
        '''
        implements a 1D and a 2D version of action value iteration algorithm
        Parameters:
            - states: all different possible states [numpy.array]
            - actions: all possible actions to do [numpy.array]
            - rewards: all different rewards for the different states - same shape as 'states'! [numpy.array]
            - terminal_states: all states which indicate 'the end of the search' [numpy.array]
            - probabilities: probabilities to do the desired - respective one of the other - action [numpy.array]
            - sequence: sequence of actions that shall be done for training [numpy.array]
            - transition: dictionary that translates the sequence into the action [Dictionary]
            - gamma: rate of forgetfulness - usually something like 0.9 [Float]
            - alpha: learning rate - usually something like (1/len(actions)) [Float]
            - verbose: how detailed the train process shall be documented. Possible values are [Integer]
                - 0 -> no information (default)
                - 1 -> more detailed information like Q(s,a) and V(s)
        Returns:
            - pi: trained policy [numpy.array]
        '''
        ## Formula for Q-Learning: q(s,a) = q(s,a) + a [r + y * max( q(s',a') ) - q(s,a)]
        ## check the shape of the states to determine whether 1D or 2D problem
        if len(states.shape) == 1:
            pi = self.q_learning_1D(states, actions, rewards, terminal_states, sequence, transition, gamma, alpha, verbose)
        elif len(states.shape) == 2:
            pi = self.q_learning_2D(states, actions, rewards, terminal_states, sequence, transition, gamma, alpha, verbose)
        return pi        
    
class hmm:
    '''
    class that implements Hidden Markov Models, including:
        - Markov Processes to estimate the probability of a given sequence
        - forward + backward procedure for probability estimation
        - Viterbi-Algorithm for optimal state sequence estimation
        - model-reestimation for calculating the optimal parameters
    '''
    
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