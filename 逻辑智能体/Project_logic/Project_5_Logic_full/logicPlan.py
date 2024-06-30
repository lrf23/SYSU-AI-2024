# logicPlan.py
# ------------
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
# March, 2022

"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

#______________________________________________________________________________
# QUESTION 1

def sentence1():
    """Returns a Expr instance that encodes that the following expressions are all true.

    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    sen1 = logic.disjoin(A, B)
    sen2 = ~A % (~B | C)
    sen3 = logic.disjoin(~A, ~B, C)
    return logic.conjoin(sen1, sen2, sen3)
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def sentence2():
    """Returns a Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    sen1 = C % (B | D)
    sen2 = A >> (~B & ~D)
    sen3 = ~(B & ~C) >> A
    sen4 = ~D >> C
    return logic.conjoin(sen1, sen2, sen3, sen4)
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def sentence3():
    """Using the symbols PacmanAlive[1], PacmanAlive[0], PacmanBorn[0], and PacmanKilled[0],
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    "*** BEGIN YOUR CODE HERE ***"
    a = logic.PropSymbolExpr("PacmanAlive[0]")
    b = logic.PropSymbolExpr("PacmanAlive[1]")
    c = logic.PropSymbolExpr("PacmanBorn[0]")
    d = logic.PropSymbolExpr("PacmanKilled[0]")
    sen1 = b % ((a & ~d) | (~a & c))
    sen2 = ~(a & c)
    sen3 = c
    return logic.conjoin(sen1, sen2, sen3)
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are
    sorted before converting the model to a string.

    model: Either a boolean False or a dictionary of Expr symbols (keys)
    and a corresponding assignment of True or False (values). This model is the output of
    a call to pycoSAT.
    """
    if model == False:
        return "False"
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** BEGIN YOUR CODE HERE ***"
    a = logic.to_cnf(sentence)
    b = logic.pycoSAT(a)
    if str(b) == "FALSE":
        return False
    return b
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals):
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single
    Expr instance in CNF (conjunctive normal form) that represents the logic
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(logic.pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(logic.pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(logic.pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    return logic.disjoin(literals) # 逻辑 或 操作
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def atMostOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in
    CNF (conjunctive normal form) that represents the logic that at most one of
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # Way 1
    # conjunctions = []
    # for literal in literals:
    #     not_literal = ~literal
    #     # Disjoin literal with NOT(literal) for every other element besides this literal
    #     # and add it to the list to be conjoined
    #     reached_literal = False
    #     for inner_literal in literals:
    #         if (reached_literal):
    #             not_inner_literal = ~inner_literal
    #             disjunction = logic.disjoin(not_literal, not_inner_literal)
    #             conjunctions.append(disjunction)
    #         if literal == inner_literal:
    #             reached_literal = True
    #
    # return logic.conjoin(conjunctions)

    # Way 2
    conjunctions = []
    for combination in itertools.combinations(range(len(literals)), 2):
        index1 = combination[0]
        index2 = combination[1]
        disjunction = logic.disjoin(~literals[index1], ~literals[index2])
        conjunctions.append(disjunction)
    return logic.conjoin(conjunctions)
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def exactlyOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in
    CNF (conjunctive normal form)that represents the logic that exactly one of
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # Way 1
    # conjunctions = []
    # one_must_be_true_list = []
    # for literal in literals:
    #     not_literal = ~literal
    #     one_must_be_true_list.append(literal)
    #
    #     # Disjoin literal with NOT(literal) for every other element besides this literal
    #     # and add it to the list to be conjoined
    #     reached_literal = False
    #     for inner_literal in literals:
    #         if (reached_literal):
    #             not_inner_literal = ~inner_literal
    #             disjunction = logic.disjoin(not_literal, not_inner_literal)
    #             conjunctions.append(disjunction)
    #         if literal == inner_literal:
    #             reached_literal = True
    #
    # # Add the expression that states at least one of the literals must be true
    # one_must_be_true = logic.disjoin(one_must_be_true_list)
    # conjunctions.append(one_must_be_true)
    # return logic.conjoin(conjunctions)

    # Way 2
    return logic.conjoin(atMostOne(literals), atLeastOne(literals))
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, time = parsed
            plan[int(time)] = action # Ordered by time
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    possibilities = []
    if not walls_grid[x][y+1]:
        possibilities.append( PropSymbolExpr(var_str, x, y+1, t-1)
                            & PropSymbolExpr('South', t-1))
    if not walls_grid[x][y-1]:
        possibilities.append( PropSymbolExpr(var_str, x, y-1, t-1)
                            & PropSymbolExpr('North', t-1))
    if not walls_grid[x+1][y]:
        possibilities.append( PropSymbolExpr(var_str, x+1, y, t-1)
                            & PropSymbolExpr('West', t-1))
    if not walls_grid[x-1][y]:
        possibilities.append( PropSymbolExpr(var_str, x-1, y, t-1)
                            & PropSymbolExpr('East', t-1))

    if not possibilities:
        return None

    return PropSymbolExpr(var_str, x, y, t) % disjoin(possibilities)


def pacmanSLAMSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    moved_tm1_possibilities = []
    if not walls_grid[x][y+1]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x, y+1, t-1)
                            & PropSymbolExpr('South', t-1))
    if not walls_grid[x][y-1]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x, y-1, t-1)
                            & PropSymbolExpr('North', t-1))
    if not walls_grid[x+1][y]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x+1, y, t-1)
                            & PropSymbolExpr('West', t-1))
    if not walls_grid[x-1][y]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x-1, y, t-1)
                            & PropSymbolExpr('East', t-1))

    if not moved_tm1_possibilities:
        return None

    moved_tm1_sent = conjoin([~PropSymbolExpr(var_str, x, y, t-1) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_tm1_possibilities)])

    unmoved_tm1_possibilities_aux_exprs = [] # merged variables
    aux_expr_defs = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, t - 1)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, t - 1)
        unmoved_tm1_possibilities_aux_exprs.append(wall_dir_combined_literal)
        aux_expr_defs.append(wall_dir_combined_literal % wall_dir_clause)

    unmoved_tm1_sent = conjoin([
        PropSymbolExpr(var_str, x, y, t-1),
        disjoin(unmoved_tm1_possibilities_aux_exprs)])

    return conjoin([PropSymbolExpr(var_str, x, y, t) % disjoin([moved_tm1_sent, unmoved_tm1_sent])] + aux_expr_defs)

#______________________________________________________________________________
# QUESTION 3

def pacphysics_axioms(t, all_coords, non_outer_wall_coords):
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in) 除了四方形的边所在的位置的其它所有(x, y)位置，因为pacman不能在边上
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action of DIRECTIONS at timestep t.
    """
    pacphysics_sentences = []
    "*** BEGIN YOUR CODE HERE ***"
    sen1_clauses = []
    for coords in all_coords:
        sen1_clauses.append(PropSymbolExpr(wall_str, coords[0], coords[1]) >> ~PropSymbolExpr(pacman_str, coords[0], coords[1], t))
    pacphysics_sentences.append(conjoin(sen1_clauses))
    sen2_clauses = [PropSymbolExpr(pacman_str, coords[0], coords[1], t) for coords in non_outer_wall_coords]
    pacphysics_sentences.append(exactlyOne(sen2_clauses))
    sen3_clauses = [PropSymbolExpr(action, t) for action in DIRECTIONS]
    pacphysics_sentences.append(exactlyOne((sen3_clauses)))
    "*** END YOUR CODE HERE ***"
    return conjoin(pacphysics_sentences)
    raise NotImplementedError


def check_location_satisfiability(x1_y1, x0_y0, action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - problem = An instance of logicAgents.LocMapProblem
    Return:
        - a model proving whether Pacman is at (x1, y1) at time t = 1
        - a model proving whether Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"
    # We know pacman is currently at (x0, y0) at time t = 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, 0))
    # We add pacphysics_axioms at time t = 0 and t = 1
    KB.append(pacphysics_axioms(0, all_coords, non_outer_wall_coords))
    KB.append(pacphysics_axioms(1, all_coords, non_outer_wall_coords))
    # We know pacman take action0 at time t = 0 and action1 at t = 1
    KB.append(PropSymbolExpr(action0, 0))
    KB.append(PropSymbolExpr(action1, 1))
    # We know pacman allLegalSuccessorAxioms at t+1 when t = 0
    KB.append(allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords))

    q = PropSymbolExpr(pacman_str, x1, y1, 1) # query
    # We find model 1 via KB & ~q, find model 2 via KB & q
    model1 = findModel(conjoin(KB) & ~q)
    model2 = findModel(conjoin(KB) & q)
    return (model1, model2)
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2),
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str,x0,y0,0))
    for t in range(50):
        print(t)
        pos_k=[]
        for pos in non_wall_coords:
            pos_k.append(PropSymbolExpr(pacman_str,pos[0],pos[1],t))
        KB.append(exactlyOne(pos_k))
        res1=findModel(conjoin(KB) & PropSymbolExpr(pacman_str,xg,yg,t))#判断能不能找到一个满意的赋值，并不是说KB能否推出q
        if res1:
            return extractActionSequence(res1,actions)
        act_k=[]
        for act in actions:
            act_k.append(PropSymbolExpr(act,t))
        KB.append(exactlyOne(act_k))
        for x,y in non_wall_coords:
            KB.append(pacmanSuccessorStateAxioms(x,y,t+1,walls,pacman_str))
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 5

def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    #locations = list(filter(lambda loc : loc not in walls_list, all_coords))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []
    for f in food:
        KB.append(PropSymbolExpr(food_str,f[0],f[1],0)) #0时刻这些地方有食物
    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str,x0,y0,0))
    for t in range(50):
        print(t)
        pos_k=[]
        for pos in non_wall_coords:
            pos_k.append(PropSymbolExpr(pacman_str,pos[0],pos[1],t))
        KB.append(exactlyOne(pos_k))
        food_k=[]
        for x,y in food:
            food_k.append(~PropSymbolExpr(food_str,x,y,t))
        goal=conjoin(food_k)
        res1=findModel(conjoin(KB) & goal)#判断能不能找到一个满意的赋值，并不是说KB能否推出q
        if res1:
            return extractActionSequence(res1,actions)
        act_k=[]
        for act in actions:
            act_k.append(PropSymbolExpr(act,t))
        KB.append(exactlyOne(act_k))
        for x,y in non_wall_coords:
            KB.append(pacmanSuccessorStateAxioms(x,y,t+1,walls,pacman_str))
        for x,y in food:
            expr1=PropSymbolExpr(food_str,x,y,t)& (~PropSymbolExpr(pacman_str,x,y,t))
            expr2=PropSymbolExpr(food_str,x,y,t+1)
            KB.append(expr1 % expr2)

    raise NotImplementedError
    "*** END YOUR CODE HERE ***"


# Helpful Debug Method
def visualize_coords(coords_list, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualize_bool_array(bool_arr, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)


def sensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def four_bit_percept_rules(t, percepts):
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def num_adj_walls_percept_rules(t, percepts):
    """
    SLAM uses a weaker num_adj_walls sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    num_adj_walls = sum(percepts)
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t, walls_grid, non_outer_wall_coords):
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t, walls_grid, non_outer_wall_coords):
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSLAMSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# QUESTION 6


def localization(problem, agent):
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    debug = False

    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    possible_locs_by_timestep = []
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    for x,y in all_coords:
        if (x,y) in walls_list:
            KB.append(PropSymbolExpr(wall_str,x,y))
        else:
            KB.append(~PropSymbolExpr(wall_str,x,y))
    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t,all_coords,non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t],t))
        KB.append(sensorAxioms(t,non_outer_wall_coords))
        KB.append(four_bit_percept_rules(t,agent.getPercepts()))
        pos_loc_t=[]
        for x,y in non_outer_wall_coords:
            q=PropSymbolExpr(pacman_str,x,y,t)
            res1=findModel(conjoin(KB)& ~q)
            res2=findModel(conjoin(KB) & q)
            if not res1:
                KB.append(PropSymbolExpr(pacman_str,x,y,t))
            elif not res2:
                KB.append(~PropSymbolExpr(pacman_str,x,y,t))

            if res2:
                pos_loc_t.append((x,y))
        possible_locs_by_timestep.append(pos_loc_t)
        agent.moveToNextState(agent.actions[t])
        KB.append(allLegalSuccessorAxioms(t+1,walls_grid,non_outer_wall_coords))
    return possible_locs_by_timestep

    raise NotImplementedError
    "*** END YOUR CODE HERE ***"
    return possible_locs_by_timestep

#______________________________________________________________________________
# QUESTION 7
def mapping(problem, agent):
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    #map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]
    known_map_by_timestep = []

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))#讲外围有墙作为公共知识
    KB.append(PropSymbolExpr(pacman_str,pac_x_0,pac_y_0,0))
    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t,all_coords,non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t],t))
        KB.append(sensorAxioms(t,non_outer_wall_coords))
        KB.append(four_bit_percept_rules(t,agent.getPercepts()))
        for x,y in non_outer_wall_coords:
            q=PropSymbolExpr(wall_str,x,y)
            res1=findModel(conjoin(KB)& ~ q)
            res2=findModel(conjoin(KB) & q)
            if not res1:
                KB.append(PropSymbolExpr(wall_str,x,y,t))
                known_map[x][y]=1
            elif not res2:
                KB.append(~PropSymbolExpr(wall_str,x,y,t))
                known_map[x][y]=0
            else:
                known_map[x][y]=-1
            # if res1 and res2:
            #     known_map[x][y]=-1
            # elif res1:
            #     known_map[x][y]=0
            # elif res2:
            #     known_map[x][y]=1
        known_map_by_timestep.append(copy.deepcopy(known_map))
        agent.moveToNextState(agent.actions[t])
        wall_grid=[[] for _ in range(len(known_map))]
        for x in range(len(known_map)):
            for y in range(len(known_map[x])):
                if known_map[x][y]==-1:
                    wall_grid[x].append(0)
                else:
                    wall_grid[x].append(known_map[x][y])
        KB.append(allLegalSuccessorAxioms(t+1,wall_grid,non_outer_wall_coords))

    return known_map_by_timestep
    "*** BEGIN YOUR CODE HERE ***"
    raise NotImplementedError
    "*** END YOUR CODE HERE ***"
    return known_map_by_timestep

#______________________________________________________________________________
# QUESTION 8

def slam(problem, agent):
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]
    known_map_by_timestep = []
    possible_locs_by_timestep = []

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str,pac_x_0,pac_y_0,0))
    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t,all_coords,non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t],t))
        KB.append(SLAMSensorAxioms(t,non_outer_wall_coords))
        KB.append(num_adj_walls_percept_rules(t,agent.getPercepts()))
        for x,y in non_outer_wall_coords:
            q=PropSymbolExpr(wall_str,x,y)
            res1=findModel(conjoin(KB)& ~ q)
            res2=findModel(conjoin(KB) & q)
            if not res1:
                KB.append(PropSymbolExpr(wall_str,x,y,t))
                known_map[x][y]=1
            elif not res2:
                KB.append(~PropSymbolExpr(wall_str,x,y,t))
                known_map[x][y]=0
            else:
                known_map[x][y]=-1
        pos_loc_t=[]
        for x,y in non_outer_wall_coords:
            q=PropSymbolExpr(pacman_str,x,y,t)
            res1=findModel(conjoin(KB)& ~q)
            res2=findModel(conjoin(KB) & q)
            if not res1:
                KB.append(PropSymbolExpr(pacman_str,x,y,t))
            elif not res2:
                KB.append(~PropSymbolExpr(pacman_str,x,y,t))

            if res2:
                pos_loc_t.append((x,y))
        possible_locs_by_timestep.append(pos_loc_t)
        known_map_by_timestep.append(copy.deepcopy(known_map))
        agent.moveToNextState(agent.actions[t])
        wall_grid=[[] for _ in range(len(known_map))]
        for x in range(len(known_map)):
            for y in range(len(known_map[x])):
                if known_map[x][y]==-1:
                    wall_grid[x].append(0)
                else:
                    wall_grid[x].append(known_map[x][y])
        KB.append(SLAMSuccessorAxioms(t+1,wall_grid,non_outer_wall_coords))

    return known_map_by_timestep,possible_locs_by_timestep
    "*** END YOUR CODE HERE ***"
    return known_map_by_timestep, possible_locs_by_timestep

# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
