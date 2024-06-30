# multiAgents.py
# --------------
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
# February, 2022


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function. 在每个决策点，通过一个状态评估函数分析它的可能行动来做决定

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.
        getAction 根据评估函数选择最佳的行动
        getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        返回 {NORTH, SOUTH, WEST, EAST, STOP} 中的一个行动
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best 如果有多个最佳行动（分数相同），随机选一个

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # currentGameState、successorGameState的内部内容和函数可以查看pacman.py里的gameState类
        successorGameState = currentGameState.generatePacmanSuccessor(action) # 在当前状态后采取一个行动后到达的状态
        newPos = successorGameState.getPacmanPosition() # 下一个状态的位置 （x，y）
        newFood = successorGameState.getFood() # 下一个状态时环境中的食物情况 (TTTFFFFFT......)
        newGhostStates = successorGameState.getGhostStates() # 可以查看game.py里的AgentState类的函数
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # 吃了大的白色食物（能量点）后，白幽灵的剩余持续时间

        "*** YOUR CODE HERE 可更改return的东西,即用非.getScore()的方法评估下一个状态的好坏***"
        foods=newFood.asList()
        score=0.0
        
        if len(foods)!=0:
            min_dis=min([manhattanDistance(food,newPos) for food in foods])
            score+=9/min_dis
        #score+=sum(newScaredTimes)
        #GhostsPos=newGhostStates.getPosition()

        
        if (len(newGhostStates)!=0):
            min_dis=min([manhattanDistance(gp.configuration.pos,newPos) for gp in newGhostStates])
            if (min_dis==0):
                score-=0
            else:
                score=score-(10.0/min_dis)

        return score+successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. 根据当前的游戏状态,返回一个根据minimax值选的最佳行动

        Here are some method calls that might be useful when implementing minimax.
        以下的一些函数调用可能会对你有帮助
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent 返回一个agent（包括吃豆人和幽灵）合法行动（如不能往墙的地方移动）的列表
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action 一个agent采取行动后，生成的新的游戏状态

        gameState.getNumAgents():
        Returns the total number of agents in the game 获取当前游戏中所有agent的数量

        gameState.isWin():
        Returns whether or not the game state is a winning state 判断一个游戏状态是不是目标的胜利状态

        gameState.isLose():
        Returns whether or not the game state is a losing state 判断一个游戏状态是不是游戏失败结束的状态

        提示：你可以在函数内部定义新的函数，比如 def min_value(...):
        """
        "*** YOUR CODE HERE ***"
        max_value=-float('inf')
        best_action=None
        for action in gameState.getLegalActions(agentIndex=0):
            val=self.getValue(gameState.generateSuccessor(agentIndex=0,action=action),1,0)
            if val>max_value:
                max_value=val
                best_action=action
        return best_action
    
    def getValue(self,gameState,agentIndex,depth):
        nextActions=gameState.getLegalActions(agentIndex=agentIndex)
        if (len(nextActions)==0):
            return self.evaluationFunction(gameState)
        else:
            if agentIndex==0:
                depth+=1
                if depth==self.depth:
                    return self.evaluationFunction(gameState)
                else:
                    return self.max_val(gameState,agentIndex,depth)
            else:
                return self.min_val(gameState,agentIndex,depth)
        
    def max_val(self,gameState,agentIndex,depth):
        max_val=-float('inf')
        nextActions=gameState.getLegalActions(agentIndex=agentIndex)
        for action in nextActions:
            val=self.getValue(gameState.generateSuccessor(agentIndex=agentIndex,action=action),(agentIndex+1)%gameState.getNumAgents(),depth)
            if val is not None and val>max_val:
                max_val=val
        return max_val
    
    def min_val(self,gameState,agentIndex,depth):
        min_val=float('inf')
        nextActions=gameState.getLegalActions(agentIndex=agentIndex)
        for action in nextActions:
            val=self.getValue(gameState.generateSuccessor(agentIndex=agentIndex,action=action),(agentIndex+1)%gameState.getNumAgents(),depth)
            if val is not None and val<min_val:
                min_val=val
        return min_val
    


        
        


