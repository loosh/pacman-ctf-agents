# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions
import game

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = True

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
WEIGHT_PATH = 'weights_MY_TEAM.json'

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
# [!] TODO

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='ImprovedOffensiveAgent', second='DynamicDefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class OffensiveChaseAgent(CaptureAgent):
    """
    A specialized agent that focuses on chasing pellets in a capture-the-flag Pacman game.
    """
    
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

    def chooseAction(self, gameState):
        """
        Picks among actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns features used for evaluating actions.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()    
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        """
        Returns weights for the features used in the evaluation.
        """
        return {'successorScore': 100, 'distanceToFood': -1}

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

class DefensiveReflexAgent(CaptureAgent):
  """
  A defensive agent that focuses on protecting the team's pellets.
  """

  def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.start = gameState.getAgentPosition(self.index)
      self.foodToDefend = self.getFoodYouAreDefending(gameState).asList()

  def chooseAction(self, gameState):
      actions = gameState.getLegalActions(self.index)
      bestAction = None
      bestValue = -float('inf')
      for action in actions:
          value = self.evaluate(gameState, action)
          if value > bestValue:
              bestValue = value
              bestAction = action
      return bestAction

  def evaluate(self, gameState, action):
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      return features * weights

  def getFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in opponents if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
          features['invaderDistance'] = min(dists)
      else:
          # Stay close to the food being defended if no invaders are visible
          dists = [self.getMazeDistance(myPos, food) for food in self.foodToDefend]
          features['distanceToFood'] = min(dists)

      return features

  def getWeights(self, gameState, action):
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'distanceToFood': -1}

  def getSuccessor(self, gameState, action):
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):
          return successor.generateSuccessor(self.index, action)
      else:
          return successor

class ImprovedOffensiveAgent(OffensiveChaseAgent):
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.positionsVisited = set()

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        # Penalize revisiting the same position to encourage exploration
        if myPos in self.positionsVisited:
            features['repeatPositionPenalty'] = 1
        else:
            features['repeatPositionPenalty'] = 0
            self.positionsVisited.add(myPos)
        
        # Clear the visited positions if it gets too large to avoid memory issues
        if len(self.positionsVisited) > 100:
            self.positionsVisited.clear()

        # Better baselines will avoid defenders!
        enemies = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            minGhostDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]) + 1
            features['distanceToGhost'] = minGhostDistance

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        weights = super().getWeights(gameState, action)
        # Ensure ghostDistanceWeight is a numeric value
        ghostDistanceWeight = 50
        repeatPositionPenaltyWeight = -100  # Assuming this is always a numeric value
        stop = -100

        weights.update({
            'distanceToGhost': ghostDistanceWeight,
            'repeatPositionPenalty': repeatPositionPenaltyWeight,
            'stop': stop,
            'reverse': -2,
        })

        # Debugging: Check that all weights are numeric
        for key, value in weights.items():
            assert isinstance(value, (int, float)), f"Weight for {key} is not numeric: {value}"

        return weights



class DynamicDefensiveAgent(DefensiveReflexAgent):
    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Determines if the agent is scared
        isScared = myState.scaredTimer > 0
        features['isScared'] = isScared

        # Identifies invaders
        invaders = [successor.getAgentState(i) for i in self.getOpponents(successor) if successor.getAgentState(i).isPacman and successor.getAgentState(i).getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            # Adjust behavior based on scared state
            if isScared:
                features['invaderDistance'] = max(dists)
            else:
                features['invaderDistance'] = min(dists)
        else:
            features['invaderDistance'] = 0

        # Encourage moving forward to a strategic position on the field
        boundaryPosition = self.getBoundaryPosition(gameState)
        features['forwardMovement'] = self.getMazeDistance(myPos, boundaryPosition)

        return features

    def getWeights(self, gameState, action):
        # Determine if the agent is scared directly here
        isScared = gameState.getAgentState(self.index).scaredTimer > 0

        # Use conditional expression to set invaderDistance weight
        invaderDistanceWeight = -15 if not isScared else 15

        # Return a dictionary of weights with numeric values
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': invaderDistanceWeight,
            'stop': -100,
            'reverse': -2,
            'isScared': 100,
            'forwardMovement': -5
        }

    def getBoundaryPosition(self, gameState):
        """
        Calculate a strategic forward position to patrol.
        This will be a position near the center of your side of the game board.
        """
        mapWidth = gameState.data.layout.width
        mapHeight = gameState.data.layout.height
        if self.red:
            x = (mapWidth // 2) - 2
        else:
            x = (mapWidth // 2) + 1

        # Look for a position without a wall, starting from the center of the map
        for y in range(mapHeight // 2, 0, -1):
            if not gameState.hasWall(x, y):
                return (x, y)
        for y in range(mapHeight // 2, mapHeight):
            if not gameState.hasWall(x, y):
                return (x, y)

        return self.start  # Fallback to the start position if no suitable position is found

