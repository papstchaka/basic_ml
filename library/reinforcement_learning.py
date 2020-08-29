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