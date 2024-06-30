# search.py
# ---------
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

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    result=[]
    st_state=(problem.getStartState(),[])
    frontier=util.Stack()
    frontier.push(st_state)
    expanded=[]
    while not frontier.isEmpty():
        (node,path)=frontier.pop()
        if problem.isGoalState(node):
            result=path
            break
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                new_path=path+[child[1]]
                new_node=(child[0],new_path)
                frontier.push(new_node)

    return result


    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    result=[]
    st_state=(problem.getStartState(),[])
    frontier=util.Queue()
    frontier.push(st_state)
    expanded=[st_state[0]]
    success=False
    while not frontier.isEmpty():

        (node,path)=frontier.pop()
        if (problem.isGoalState(node)):
            result=path
            break
        #expanded.append(node)
        for child in problem.getSuccessors(node):
            if child[0] not in expanded:
                new_path=path+[child[1]]
                frontier.push((child[0],new_path))
                expanded.append(child[0])
                

            
    return result
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    result=[]
    st_state=(problem.getStartState(),[])
    frontier=util.PriorityQueue()
    frontier.push(st_state,0)
    expanded=[]
    while not frontier.isEmpty():
        (node,path)=frontier.pop()
        if problem.isGoalState(node):
            result=path
            break
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                new_path=path+[child[1]]
                new_cost=problem.getCostOfActions(new_path)
                new_node=(child[0],new_path)
                frontier.push(new_node,new_cost)

    return result



    #util.raiseNotDefined()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
