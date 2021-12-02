# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
# <<<<<<< HEAD
#         "*** YOUR CODE HERE ***"
#         #initialize V_0(s) = 0
#        V_i = 0
#         for i in range(self.iterations):
#             Q_arr = []
#             for state in self.mdp.getStates():
                
#                 V_i = max(Q_arr)
    
#         # v_(k+1) = max ()
=======
        "*** YOUR CODE HERE ***"        
        for i in range(self.iterations):
            q_vals = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    q_vals[state] = 0 
                else:
                    best_action = self.getAction(state)
                    q_vals[state] = self.getQValue(state, best_action)
            self.values = q_vals
# >>>>>>> db637ad7fb3ccfae87c3626ba9bcc7994f30ed16


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #weighted sum
        #probability of occurance * utility of going s->s'
        #for all possible s' ?
        q_val = 0
        pairs_list = self.mdp.getTransitionStatesAndProbs(state, action)
        for pair in pairs_list:
            possible_state = pair[0]
            prob = pair[1]
            if (self.getValue(possible_state) != None):
                reward = self.mdp.getReward(state, action, possible_state)
                q_val += prob * (reward + (self.discount * self.getValue(possible_state)))
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        action_dict = {}
        for action in self.mdp.getPossibleActions(state):
<<<<<<< HEAD
# <<<<<<< HEAD
#             #fill this in later
#             #for now just do a random whatever
#             action_dict[action] = random.choice([1,2,3,4,5])
# =======
            action_dict[action] = self.computeQValueFromValues(state, action)
# >>>>>>> db637ad7fb3ccfae87c3626ba9bcc7994f30ed16
=======
            action_dict[action] = self.getQValue(state, action)
>>>>>>> b2e2c503814e52a1e8a7d3de8df00254402d66cd
        return max(action_dict, key= lambda x: action_dict[x])
        


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        #initializing the values passed into this constructor exactly the same as in question 1
        # self.mdp = mdp
        # self.discount = discount 
        # self.iterations = iterations

        # self.values = util.Counter() # A Counter is a dict with default 0
        # self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
<<<<<<< HEAD
        list_states = self.mdp.getStates()
        vals = self.values.copy()
        for i in range(self.iterations):
            state = list_states[i%len(list_states)]
            if self.mdp.isTerminal(state):
                continue
            else:
                action = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, action)
# =======
#         """
#         from project spec:
#             In the first iteration, only update the value of the first state in the states list. 
#             In the second iteration, only update the value of the second. 
#             Keep going until you have updated the value of each state once, 
#             then start back at the first state for the subsequent iteration. 
#             If the state picked for updating is terminal, nothing happens in that iteration.
#         """
#         num_states = sum(self.values.itervalues())
#         vals = self.values.copy()
#         for i in range(self.iterations):
#             state = vals.__getitem__(i%num_states)
#             if self.mdp.isTerminal(state):
#                 continue
#             else:
#                 action = self.getAction(state)
#                 vals[state] = self.getQValue(state, action)
#         self.values = vals

# >>>>>>> b2e2c503814e52a1e8a7d3de8df00254402d66cd

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    #helper function to compute predecessors of a state
    #returns a set -- avoids duplicates
    def computePredecessors(self, curr_state):
        predecessors = set()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                pairs_list = self.mdp.getTransitionStatesAndProbs(state, action)
                for pair in pairs_list:
                    possible_state = pair[0]
                    prob = pair[1]
                    if (possible_state == curr_state) and (prob > 0):
                        #state is a predecessor of curr_state!
                        predecessors.add(state)
                        break
                else:
                    continue
                break
        return predecessors


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #compute predecessors of all states
        """
        First, we define the predecessors of a state s as all states 
        that have a nonzero probability of reaching s by taking some action a.
        """
        predecessor_dict = {}
        for s in self.mdp.getStates():
            #key: state
            #value: set of all possible predecessors
            predecessor_dict[s] = self.computePredecessors(s)

        #initialize empty priority queue
        queue = util.PriorityQueue()
        #for each non-terminal state s, do:
        for s in self.mdp.getStates():
            #find the absolute value of the diff between current value of s in self.values 
            #and the highest q value across all possible actions from s
            best_action = self.getAction(s)
            highest_q = self.getQValue(s, best_action)
            diff = abs(self.getValue(s) - highest_q)
            #push s to the queue with priority negative diff
            queue.update(s, -diff)
        #for iteration in 0, 1, 2, ..., self.iterations-1, do:
        for i in range(self.iterations):
            #if the priority queue is empty, terminate
            if queue.isEmpty():
                break
            #pop a state off the priority queue
            s = queue.pop()
            #update the value of s (if it is not a terminal state) in self.values
            if not self.mdp.isTerminal(s):
                best_action = self.getAction(s)
                self.values[s] = self.getQValue(s, best_action)
            #for each predecessor p of s do:
            for p in predecessor_dict[s]:
                #find the absolute value of the diff between current value of p in self.values 
                #and the highest q value across all possible actions from p
                best_action = self.getAction(p)
                highest_q = self.getQValue(p, best_action)
                diff = abs(self.getValue(p) - highest_q)
                #if diff > theta, push p directly into the priority queue with priority -diff
                if (diff > self.theta):
                    queue.update(p, -diff)

