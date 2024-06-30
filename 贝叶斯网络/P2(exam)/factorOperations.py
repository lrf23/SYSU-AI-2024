# factorOperations.py
# -------------------
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

from bayesNet import Factor
import operator as op
import util
import functools

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions: （具体可以查看bayesNet.py里的Factor类）
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion 测试输入是否正确
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    '''
    Factor() Takes in as input an iterable unconditionedVariables, an iterable 
        conditionedVariables, and a variableDomainsDict as a mapping from 
        variables to domains.

        inputUnconditionedVariables is an iterable of variables (represented as strings)
            that contains the variables that are unconditioned in this factor 
        inputConditionedVariables is an iterable of variables (represented as strings)
            that contains the variables that are conditioned in this factor 
        inputVariableDomainsDict is a dictionary from variables to domains of those 
            variables (typically represented as a list but could be any iterable)
    '''
    unconditioned = []
    conditioned = []
    variableDomainsDict = {}
    print(type(factors))
    print(list(factors)) #需转化为list来获取它的变量值域，不转化的话会报错

    if factors and len(factors) > 0:
        variableDomainsDict = list(factors)[0].variableDomainsDict()
        print(variableDomainsDict) # 可以假设所有输入的Factor都来自同一个BayesNet，所以它们的variableDomainsDicts都是一样的。

    # 注意观察题目的Hints的一些joinFactors()的例子
    for f in factors:
        temp_unconditioned = f.unconditionedVariables()
        temp_conditioned = f.conditionedVariables()
        unconditioned.extend(temp_unconditioned)
        for conditioned_var in temp_conditioned:
            if conditioned_var not in conditioned: # 如果输入因子的条件变量还没被加入新因子的候选条件变量集中
                conditioned.append(conditioned_var)
    conditioned = [var for var in conditioned if var not in unconditioned] # 如果新因子的候选条件变量不是无条件边量集中的一员

    # 新因子的有条件变量、无条件变量和变量值域都有了，可以初始化了
    newFactor = Factor(unconditioned, conditioned, variableDomainsDict)
    # 初始化之后，需要进一步对新因子（条件概率表）中的每行每列的条目赋值。
    assignments = newFactor.getAllPossibleAssignmentDicts()
    for assignment in assignments:
        prob = 1
        for factor in factors:
            prob *= factor.getProbability(assignment) # 概率条目是输入Factors的相应行的乘积
        newFactor.setProbability(assignment, prob)
    return newFactor
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        unconditioned = factor.unconditionedVariables()
        # 除了要被消除的变量之外的无条件变量
        unconditioned = [var for var in unconditioned if var != eliminationVariable]
        conditioned = factor.conditionedVariables()
        variableDomainsDict = factor.variableDomainsDict()
        # 新的因子的无条件变量、有条件变量和变量值域都准备好了，可以初始化了
        newFactor = Factor(unconditioned, conditioned, variableDomainsDict)
        # 获取要被消除的变量的值域，开始求和消元（对新因子的每一个条目都要操作）
        domain = variableDomainsDict[eliminationVariable]
        for assignment in newFactor.getAllPossibleAssignmentDicts():
            prob = 0
            for elim_val in domain:
                old_assignment = assignment.copy()
                old_assignment[eliminationVariable] = elim_val #对assignment中的被消除的变量的取值进行遍历
                prob += factor.getProbability(old_assignment) #对输入因子在该变量所有的赋值情况下的assignment的概率求和
            newFactor.setProbability(assignment, prob)
        return newFactor
        util.raiseNotDefined()
        "*** END YOUR CODE HERE ***"

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.


    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain (还包括无条件变量中只有一个条目的那些变量).  Since there is only one entry
    in that variable's domain, we can either assume it was assigned as evidence to have only one variable
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print("Factor failed normalize typecheck: ", factor)
            raise ValueError("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    "*** YOUR CODE HERE ***"
    variableDomainsDict = factor.variableDomainsDict()
    unconditioned = factor.unconditionedVariables()
    conditioned = factor.conditionedVariables()

    prob_sum = 0 # 归一化式子的分母
    old_assignments = factor.getAllPossibleAssignmentDicts()
    for old_assignment in old_assignments:
        prob_sum += factor.getProbability(old_assignment)
    if prob_sum == 0:
        return None

    for var in unconditioned:
        if len(variableDomainsDict[var]) == 1: # 如果无条件变量的值域只有一个（即已有赋值），则它其实也成为了一个"条件"，加入到有条件变量中
            conditioned.add(var)
    unconditioned = [var for var in unconditioned if var not in conditioned]
    newFactor = Factor(unconditioned, conditioned, variableDomainsDict)

    for assignment in newFactor.getAllPossibleAssignmentDicts():
        prob = factor.getProbability(assignment)
        newFactor.setProbability(assignment, prob / prob_sum)
    return newFactor
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

