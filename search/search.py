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
    #to return: LIST of actions that reaches the goal
    #fringe -> LIFO stack
    
    #initializing search tree
    fringe = util.Stack()
    fringe.push(problem.getStartState())

    #keep track of our path
    all_paths = util.Stack()
    all_paths.push([])

    #keep track of what states we went to already
    all_states = set()
    while (fringe.isEmpty() == False):
        #get the current state
        curr_state = fringe.pop()
        all_states.add(curr_state)

        curr_path = all_paths.pop()
        
        #did we get to the end?
        if problem.isGoalState(curr_state):
            #done !
            return curr_path

        #visit children
        for succ in problem.getSuccessors(curr_state):
            next_state = succ[0]
            # print(next_state)
            next_path = succ[1]
            # print(next_path)
            #have we visited the state before?
            if next_state in all_states:
                continue
            fringe.push(next_state)
            all_paths.push(curr_path + [next_path])
        
    #if we ended up here...we failed bestie
    return False
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #to return: LIST of actions that reaches the goal
    #fringe -> FIFO queue
    
    #initializing search tree
    fringe = util.Queue()
    # print("breadthFirstSearch called (search.py)")
    # print("passing in start state: ")
    # print(problem.getStartState)
    fringe.push(problem.getStartState())

    #keep track of our path
    all_paths = util.Queue()
    all_paths.push([])

    #keep track of what states we went to already
    all_states = set()
    # print("adding to all_states")
    all_states.add(problem.getStartState())
    while (fringe.isEmpty() == False):
        #get the current state
        # print("popping fringe")
        curr_state = fringe.pop()
        # print("curr_state:")
        # print(curr_state)

        curr_path = all_paths.pop()
        # print("curr_path:")
        # print(curr_path)
        #did we get to the end?
        # print("problem is goal state?")
        if problem.isGoalState(curr_state):
            print("true")
            #done !
            return curr_path

        #visit children
        for succ in problem.getSuccessors(curr_state):
            # print("successor")
            # print(succ)
            next_state = succ[0]
            #print(next_state)
            next_path = succ[1]
            #print(next_path)
            #have we visited the state before?
            if next_state in all_states:
                continue
            fringe.push(next_state)
            all_states.add(next_state)
            all_paths.push(curr_path + [next_path])
        
    #if we ended up here...we failed bestie
    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #to return: LIST of actions that reaches the goal
    #fringe -> priority queue
    
    #initializing search tree
    fringe = util.PriorityQueue()
    #cost is 0 bc we started here
    fringe.push(problem.getStartState(), 0)

    #keep track of our path
    all_paths = util.PriorityQueue()
    all_paths.push([], 0)

    #keep track of costs
    costs = util.PriorityQueue()
    costs.push(0, 0)

    #keep track of what states we went to already
    all_states = set()
    while (fringe.isEmpty() == False):
        #get the current state
        curr_state = fringe.pop()
        curr_path = all_paths.pop()
        curr_cost = costs.pop()
        if curr_state in all_states:
            continue
        all_states.add(curr_state)
        #did we get to the end?
        if problem.isGoalState(curr_state):
            #done !
            return curr_path

        #visit children
        for succ in problem.getSuccessors(curr_state):
            next_state = succ[0]
            #print(next_state)
            next_path = succ[1]
            #print(next_path)
            next_cost = curr_cost + succ[2]
            #have we visited the state before?
            if next_state in all_states:
                continue
            fringe.push(next_state, next_cost)
            all_paths.push(curr_path + [next_path], next_cost)
            costs.push(next_cost, next_cost)
        
    #if we ended up here...we failed bestie
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    #f(n) = g(n) + h(n) where g is the uniform cost function and i think h is the hueristic function
   
   
    # implementation based on note
    open_list = util.PriorityQueue()
    # one needs to initialize the open list
    current = problem.getStartState()

    open_list.push((current,[],0), 0 + heuristic(current, problem))

    closed_list = []

    pop = open_list.pop()
    pop_state = pop[0]
    pop_path = pop[1]
    pop_cost = pop[2]

   
    #i saw on piazza that this is how u know the algorithm should terminate
    while(not problem.isGoalState(pop_state)):
       
        if pop_state not in closed_list:
            closed_list.append(pop_state)

            # closed_list.append(q[0])
            list_of_successors = problem.getSuccessors(pop_state)
            for successor in list_of_successors:
                if successor[0] not in closed_list:
                    new_list = list(pop_path)
                    new_list.append(successor[1])

                    new_weight = successor[2] + pop_cost+ heuristic(successor[0], problem)
                    open_list.push((successor[0], new_list, successor[2] + pop_cost), new_weight)

        if not open_list.isEmpty():
            new_one= open_list.pop()
            pop_state = new_one[0]
            pop_path = new_one[1]
            pop_cost = new_one[2]

    return pop_path



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
