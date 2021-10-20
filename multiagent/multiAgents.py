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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        weight = 0 
        
        bestFar = float("inf")

        foodStates = newFood.asList()
        for each in foodStates:
            eachDistance = manhattanDistance(each, newPos)
            if eachDistance < bestFar: 
                bestFar = eachDistance

        weight = 1/bestFar


        for ghosts in newGhostStates:
            ghostp = ghosts.getPosition()
            manhattanDistance1 = manhattanDistance(ghostp, newPos)
            if (manhattanDistance1 == 0):
                return -float("inf")
            weight -= 2/(manhattanDistance1)


        weight += successorGameState.getScore()

        return weight

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
    def getMax(self, gameState, depth):
        #pacman calls this
        if ((depth == 0) or (gameState.isWin()) or (gameState.isLose())):
            return self.evaluationFunction(gameState)
        v = -math.inf
        for i in gameState.getLegalActions(0):
            v = max(v, self.getMin(gameState.generateSuccessor(0, i), depth, 1))
        return v



    def getMin(self, gameState, depth, ghostId):
        #ghosts call this
        #base case
        if ((depth == 0) or (gameState.isWin()) or (gameState.isLose())):
            return self.evaluationFunction(gameState)
        v = math.inf
        #check to see if we've gone through all the states
        numAgents = gameState.getNumAgents() - 1
        if (ghostId == numAgents):
            for i in gameState.getLegalActions(ghostId):
                #all ghosts moved, so pacman can go next -> getMax
                v = min(v, self.getMax(gameState.generateSuccessor(ghostId, i), depth - 1))
        else: 
            for i in gameState.getLegalActions(ghostId):
                #move ghosts -> getMin
                v = min(v, self.getMin(gameState.generateSuccessor(ghostId, i), depth, ghostId + 1))
        return v
        

    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        v = -math.inf
        best = Directions.STOP 
        for i in gameState.getLegalActions(0):
            currState = gameState.generateSuccessor(0, i)
            temp = v
            v = max(v, self.getMin(currState, self.depth, 1))
            if v > temp:
                best = i
        return best

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
   
    def getMax(self, gameState, depth, alpha, beta): 
        #LEC SLIDES:
        # def max-value(state, α, β):
        # initialize v = -∞
        # for each successor of state:
        # v = max(v, value(successor, α, β))
        # if v ≥ β return v
        # α = max(α, v)
        # return v
       #pacman calls this
        if ((depth == 0) or (gameState.isWin()) or (gameState.isLose())):
            return self.evaluationFunction(gameState)
        v = -math.inf
        for i in gameState.getLegalActions(0):
            v = max(v, self.getMin(gameState.generateSuccessor(0, i), depth, alpha, beta, 1))
            if (v > beta):
                return v
            alpha = max(alpha, v)
        return v

    def getMin(self, gameState, depth, alpha, beta, ghostId):
        #LEC SLIDES:
        # def min-value(state , α, β):
        # initialize v = +∞
        # for each successor of state:
        # v = min(v, value(successor, α, β))
        # if v ≤ α return v
        # β = min(β, v)
        # return v
       #ghosts call this
       #base case
        if ((depth == 0) or (gameState.isWin()) or (gameState.isLose())):
            return self.evaluationFunction(gameState)
        v = math.inf
        #check to see if we've gone through all the states
        numAgents = gameState.getNumAgents() - 1
        if (ghostId == numAgents):
            for i in gameState.getLegalActions(ghostId):
                #all ghosts moved, so pacman can go next -> getMax
                v = min(v, self.getMax(gameState.generateSuccessor(ghostId, i), depth - 1, alpha, beta))
                if (v < alpha):
                    return v
                beta  = min(v, beta)
        else: 
            for i in gameState.getLegalActions(ghostId):
                #move ghosts -> getMin
                v = min(v, self.getMin(gameState.generateSuccessor(ghostId, i), depth, alpha, beta, ghostId + 1))
                if (v < alpha):
                    return v
                beta  = min(v, beta)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = -math.inf
        best = Directions.STOP 
        alpha = -math.inf
        beta = math.inf
        for i in gameState.getLegalActions(0):
            currState = gameState.generateSuccessor(0, i)
            temp = v
            v = max(v, self.getMin(currState, self.depth, alpha, beta, 1))
            if (v > temp):
                best = i
            if (v > beta):
                return best
            alpha = max(v, alpha)
        return best
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getMax(self, gameState, depth):
        if ((depth == 0) or (gameState.isWin()) or (gameState.isLose())):
            return self.evaluationFunction(gameState)
        v = -math.inf
        for i in gameState.getLegalActions(0):
            v = max(v, self.getE_x(gameState.generateSuccessor(0, i), depth, 1))
        return v
    
    def getE_x(self, gameState, depth, ghostId):
        #expected value of the node
        if ((depth == 0) or (gameState.isWin()) or (gameState.isLose())):
            return self.evaluationFunction(gameState)
        v = 0
        #check to see if we've gone through all the states
        numAgents = gameState.getNumAgents() - 1
        if (ghostId == numAgents):
            for i in gameState.getLegalActions(ghostId):
                #all ghosts moved, so pacman can go next -> getMax
                v += self.getMax(gameState.generateSuccessor(ghostId, i), depth - 1)
        else: 
            for i in gameState.getLegalActions(ghostId):
                #move ghosts -> getE_x
                v += self.getE_x(gameState.generateSuccessor(ghostId, i), depth, ghostId + 1)
        return (v / len(gameState.getLegalActions(ghostId)))

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        v = -math.inf
        best = Directions.STOP 
        for i in gameState.getLegalActions(0):
            currState = gameState.generateSuccessor(0, i)
            temp = v
            v = max(v, self.getE_x(currState, self.depth, 1))
            if v > temp:
                best = i
        return best

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
