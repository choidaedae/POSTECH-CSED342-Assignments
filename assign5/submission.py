from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = [] #score, action

        for action in state.getLegalActions(index):
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          return min(nodes)
        
    action = compute(gameState, 0, self.depth)[1]
    return action 

    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []

        for action in state.getLegalActions(index):
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          return min(nodes)
        
    if self.index == gameState.getNumAgents()-1: #next is pacman 
      next = 0 
      nextdepth = self.depth - 1
    else: # next is ghost
      next = self.index + 1
      nextdepth = self.depth

    
    score = compute(gameState.generateSuccessor(self.index, action), next, nextdepth)[0]
    return score 
        
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []
        actions = state.getLegalActions(index)

        for action in actions:
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          E = sum(node[0] for node in nodes) / len(nodes)
          return (E, random.choice(actions))

        
    action = compute(gameState, self.index, self.depth)[1]
    return action 

    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []
        actions = state.getLegalActions(index)

        for action in actions:
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          E = sum(node[0] for node in nodes) / len(nodes)
          return (E, random.choice(actions))

  
    if self.index == gameState.getNumAgents()-1: #next is pacman 
      next = 0 
      nextdepth = self.depth - 1
    else: # next is ghost
      next = self.index + 1
      nextdepth = self.depth

    score = compute(gameState.generateSuccessor(self.index, action), next, nextdepth)[0]
    return score 

    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []
        actions = state.getLegalActions(index)
        for action in actions:
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          A = len(nodes)
          E = 0
          p_stop = 0.5 + 0.5 / A
          p_else = 0.5 / A
          distribution = []
          for node in nodes:
            if node[1] == Directions.STOP: 
              E += p_stop * node[0]
              distribution.append(p_stop)
            else: 
              E += p_else * node[0]
              distribution.append(p_else)

          return (E, random.choices(actions, weights = distribution))

      
    action = compute(gameState, 0, self.depth)[1]
    return action 
  
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []
        actions = state.getLegalActions(index)
        for action in actions:
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          A = len(nodes)
          E = 0
          p_stop = 0.5 + 0.5 / A
          p_else = 0.5 / A
          distribution = []
          for node in nodes:
            if node[1] == Directions.STOP: 
              E += p_stop * node[0]
              distribution.append(p_stop)
            else: 
              E += p_else * node[0]
              distribution.append(p_else)

          return (E, random.choices(actions, weights = distribution))

      
    if self.index == gameState.getNumAgents()-1: #next is pacman 
      next = 0 
      nextdepth = self.depth - 1
    else: # next is ghost
      next = self.index + 1
      nextdepth = self.depth

    score = compute(gameState.generateSuccessor(self.index, action), next, nextdepth)[0]
    return score 

    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []
        actions = state.getLegalActions(index)

        for action in actions:
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          if index % 2 == 1:
            return min(nodes)
          else:
            E = sum(node[0] for node in nodes) / len(nodes)
            return (E, random.choice(actions))
          
    action = compute(gameState, 0, self.depth)[1]
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth):
      
      if state.isWin() or state.isLose() or state.getLegalActions(index) == [Directions.STOP]:  #if IsEnd(s) -> getScore
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth

        nodes = []

        actions = state.getLegalActions(index)

        for action in actions:
          nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth)[0], action))

        if isAgent(index): # agent(pacman)
          return max(nodes)
        else:  # opp(ghost)
          if index % 2 == 1:
            return min(nodes)
          else:
            E = sum(node[0] for node in nodes) / len(nodes)
            return (E, random.choice(actions))
          
    if self.index == gameState.getNumAgents()-1: #next is pacman 
      next = 0 
      nextdepth = self.depth - 1
    else: # next is ghost
      next = self.index + 1
      nextdepth = self.depth

    score = compute(gameState.generateSuccessor(self.index, action), next, nextdepth)[0]
    return score 

    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth, alpha, beta):
      if state.isWin() or state.isLose() or state.getLegalActions() == [Directions.STOP]:  # IsEnd(s)
        return (state.getScore(), Directions.STOP)
      elif depth == 0:  # d = 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      else:
        if index == state.getNumAgents()-1: #next is pacman 
          next = 0 
          nextdepth = depth - 1
        else: # next is ghost
          next = index + 1
          nextdepth = depth
        actions = state.getLegalActions(index)
        if isAgent(index):  # Player(s) = a0, pacman, max node
          value = float("-inf")
          best_action = ''
          for action in actions:
            v = compute(state.generateSuccessor(index, action), next, nextdepth, alpha, beta)[0]
            if v > value:
              value = v
              best_action = action
            alpha = max(alpha, v)
            if alpha >= beta:
              break
          return (value, best_action)
        else:  # Player(s) = a1, ..., an, ghost, min node
          if index % 2 == 1:
            value = float("inf")
            best_action = ''
            for action in actions:
              v = compute(state.generateSuccessor(index, action), next, nextdepth, alpha, beta)[0]
              if v < value:
                value = v
                best_action = action
              beta = min(beta, v)
              if alpha >= beta:
                break
            return (value, best_action)
          else: 
            nodes = []
            for action in actions:
              nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth, alpha, beta)[0], action))
            E = sum(node[0] for node in nodes) / len(nodes)
            return (E, random.choice(actions))

    action = compute(gameState, 0, self.depth, float("-inf"), float("inf"))[1]
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def isAgent(index):
      return True if index == 0 else False
    
    def compute(state, index, depth, alpha, beta):
      if state.isWin() or state.isLose() or state.getLegalActions() == [Directions.STOP]:  # IsEnd(s)
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  # d = 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
      
      else:
        if index == state.getNumAgents()-1: # pacman 
          next = 0 
          nextdepth = depth - 1
        else: # ghost
          next = index + 1
          nextdepth = depth
        actions = state.getLegalActions(index)
        if isAgent(index):  
          value = float("-inf")
          best_action = ''
          for action in actions:
            v = compute(state.generateSuccessor(index, action), next, nextdepth, alpha, beta)[0]
            if v > value:
              value = v
              best_action = action
            alpha = max(alpha, v)
            if alpha >= beta:
              break
          return (value, best_action)
        else:  # Player(s) = a1, ..., an, ghost, min node
          if index % 2 == 1:
            value = float("inf")
            best_action = ''
            for action in actions:
              v = compute(state.generateSuccessor(index, action), next, nextdepth, alpha, beta)[0]
              if v < value:
                value = v
                best_action = action
              beta = min(beta, v)
              if alpha >= beta:
                break
            return (value, best_action)
          else: 
            nodes = []
            for action in actions:
              nodes.append((compute(state.generateSuccessor(index, action), next, nextdepth, alpha, beta)[0], action))
            E = sum(node[0] for node in nodes) / len(nodes)
            return (E, random.choice(actions))

    if self.index == gameState.getNumAgents()-1: #next is pacman 
      next = 0 
      nextdepth = self.depth - 1
    else: # next is ghost
      next = self.index + 1
      nextdepth = self.depth

    score = compute(gameState.generateSuccessor(self.index, action), next, nextdepth, float("-inf"), float("inf"))[0]
    return score 

    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  # Feature Extractor
  def getDistanceToNearestFood(pacmanPos):
        nearestFoodDist = float("inf")
        for foodPos in currentGameState.getFood().asList():
            foodDist = manhattanDistance(pacmanPos, foodPos)
            if foodDist < nearestFoodDist:
                nearestFoodDist = foodDist
        return nearestFoodDist # 가장 가까운 음식의 맨해튼 거리 반환 
    
  def getDistanceToNearestCapsule(pacmanPos):
        nearestCapsuleDist = float("inf")
        for capsulePos in currentGameState.getCapsules():
            capsuleDist = manhattanDistance(pacmanPos, capsulePos)
            if capsuleDist < nearestCapsuleDist:
                nearestCapsuleDist = capsuleDist
        return nearestCapsuleDist # 가장 가까운 캡슐의 맨해튼 거리 반환 
    
  def getScaredGhosts():
        scaredGhostPos = []
        for agentIndex in range(1, currentGameState.getNumAgents()):
            ghostState = currentGameState.getGhostState(agentIndex)
            if ghostState.scaredTimer > 0: scaredGhostPos.append(ghostState.getPosition())
        numScaredGhosts = len(scaredGhostPos)
        return (numScaredGhosts, scaredGhostPos) # Ghost의 숫자와 위치 정보 반환 
    
  def getDistanceToNearestScaredGhost(pacmanPos, scaredGhostPos):
        nearestScaredGhostDist = float("inf")
        for ghostPos in scaredGhostPos:
            scaredGhostDist = manhattanDistance(pacmanPos, ghostPos)
            if scaredGhostDist < nearestScaredGhostDist:
                nearestScaredGhostDist = scaredGhostDist
        return nearestScaredGhostDist # ScaredGhost의 거리 계산
    
  score = currentGameState.getScore() # Compute Score of currentstate
  pacmanPos = currentGameState.getPacmanPosition() # Pacman position
  numCapsules = len(currentGameState.getCapsules()) # Number of capsules
  numScaredGhosts, scaredGhostPos = getScaredGhosts() # Get information of ghosts  
  scoreforCapsules = 0

  if numScaredGhosts > 0:
        nearestObjectiveDist = getDistanceToNearestScaredGhost(pacmanPos, scaredGhostPos)
        scoreforCapsules = 50 * numCapsules # to prevent eating capsules
  else:
    if numCapsules > 0:
        nearestObjectiveDist = getDistanceToNearestCapsule(pacmanPos)
    else:
        nearestObjectiveDist = getDistanceToNearestFood(pacmanPos)

  weights = [1, 10., 150., 1] # get some Heurisitcs...
  features = [score, 1./nearestObjectiveDist, 1./(numCapsules+1), scoreforCapsules]

  evalScore = 0
  for w,f in zip(weights, features): # inner product weights and features 
    evalScore += w * f 

  return evalScore
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction