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
# Note: use it for educational purposes in School of Artificial Intelligence, Sun Yat-sen University.
# Lecturer: Zhenhui Peng (pengzhh29@mail.sysu.edu.cn)
# Credit to UC Berkeley (http://ai.berkeley.edu)
# April, 2022

import mdp, util

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
        self.values = util.Counter() # A Counter is a dict with default 0，可在下面的计算过程中存储{state1: value1, state2: value2, ...}
        self.runValueIteration()

    ### Question 1
    def runValueIteration(self):
        # Write value iteration code here
        '''
        functions that could be useful:
        self.mdp.getStates()
        util.Counter()
        self.mdp.getPossibleActions(state)
        self.computeQValueFromValues(state, action)
        '''
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            new_values={}
            for state in self.mdp.getStates():
                act=self.computeActionFromValues(state)
                if act:
                     new_values[state]=self.computeQValueFromValues(state, act)
            for state in self.mdp.getStates():
                if state in new_values:
                    self.values[state]=new_values[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          functions that could be useful:
          self.mdp.getTransitionStatesAndProbs(state, action)
          self.mdp.getReward(state, action, next_state)
        """
        "*** YOUR CODE HERE ***"
        T=self.mdp.getTransitionStatesAndProbs(state, action)
        l=len(T)
        res=0
        for i in range(l):
            R=self.mdp.getReward(state, action, T[i][0])
            res+=T[i][1]*(R+self.discount*self.values[T[i][0]])
        return res
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          functions that could be useful:
          self.mdp.getPossibleActions(state)
          self.computeQValueFromValues(state, action)
        """
        "*** YOUR CODE HERE ***"
        act_list=self.mdp.getPossibleActions(state)
        if len(act_list)==0:
            return None
        max_Q=-1e9
        res_act=None
        for act in act_list:
            Q=self.computeQValueFromValues(state, act)
            if Q>max_Q:
                max_Q=Q
                res_act=act
        return res_act
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
