# qlearningAgents.py
# ------------------
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
# May, 2022

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

### Question 3 and 4
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        ## Needed for Question 3
        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        ## Needed for Question 3
        if (state, action) not in self.qValues:
            return 0.0
        else:
            return float(self.qValues[(state, action)])
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        ## Needed for Question 3
        actions=self.getLegalActions(state)
        if len(actions)==0:
            return 0.0
        else:
            return max([self.getQValue(state,action) for action in actions])
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        ## Needed for Question 3
        actions=self.getLegalActions(state)
        if len(actions)==0:
            return None
        else:
            bestAction=None
            bestValue=float('-inf')
            for action in actions:
                value=self.getQValue(state,action)
                if value>bestValue:
                    bestValue=value
                    bestAction=action
                elif value==bestValue:
                    bestAction=random.choice([bestAction,action])
            return bestAction
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        ## Needed for Question 4
        if util.flipCoin(self.epsilon):#自由探索的概率
            action=random.choice(legalActions)
        else:
            action=self.computeActionFromQValues(state)
        return action
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        ## Needed for Question 3
        "*** YOUR CODE HERE ***"
        sample=reward+self.discount*self.computeValueFromQValues(nextState)
        self.qValues[(state,action)]=(1-self.alpha)*self.getQValue(state,action)+self.alpha*sample

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

## Question 5 No new implementation needed.
class PacmanQAgent(QLearningAgent):

    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

### Question 6
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features=self.featExtractor.getFeatures(state,action)
        qValue=0
        for feature in features:
            qValue+=self.weights[feature]*features[feature]
        return qValue
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features=self.featExtractor.getFeatures(state,action)
        difference=(reward+self.discount*self.computeValueFromQValues(nextState))-self.getQValue(state,action)
        for feature in features:
            self.weights[feature]+=self.alpha*difference*features[feature]
        #util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights)
