from captureAgents import CaptureAgent
import random, util
from game import Directions
import numpy as np
from util import nearestPoint



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
               first='QLearningOffensiveAgent', second='QLearningDefensiveAgent'):
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

class FoundationAgent(CaptureAgent):
    
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return features * self.weights
    
    def update(self, gameState, action, nextState, reward):
        correction = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(gameState, action)
        features = self.getFeatures(gameState, action)
        for feature in features:
            self.weights[feature] += self.alpha * correction * features[feature]
    
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        bestValue = -float("inf")
        
        for action in actions:
            nextState = self.getSuccessor(gameState, action)
            if isinstance(self, QLearningOffensiveAgent):
                reward = self.getOffensiveReward(gameState, action, nextState)
            else:  # Assuming QLearningDefensiveAgent
                reward = self.getDefensiveReward(gameState, action, nextState)
            self.update(gameState, action, nextState, reward)
            qValue = self.getQValue(gameState, action)
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action

        return bestAction

    def getValue(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if len(actions) == 0:
            return 0.0
        return max(self.getQValue(gameState, action) for action in actions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


class QLearningOffensiveAgent(FoundationAgent):
    """
    Offensive Q-Learning agent.
    """
    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8, **kwargs):
        super().__init__(index, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.weights = util.Counter({
            'successorScore': 100,
            'distanceToFood': -1,
            'distanceToGhost': 2,
            'stop': -100,
            'reverse': -2,
            'repeatPositionPenalty': -100,
        })
        self.positionsVisited = set()  # Keep track of visited positions
    
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Distance to the nearest food
        foodList = self.getFood(successor).asList()    
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        
        # Penalize revisiting the same position
        if successor.getAgentState(self.index).getPosition() in self.positionsVisited:
            features['repeatPositionPenalty'] = 1
        else:
            self.positionsVisited.add(successor.getAgentState(self.index).getPosition())

        # Avoid stopping and reverse movements
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Ghost avoidance
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            minGhostDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
            features['distanceToGhost'] = minGhostDistance
        
        return features

    def getOffensiveReward(self, gameState, action, nextState):
        reward = 0
        if self.getScore(nextState) - self.getScore(gameState) > 0:
            reward += 100  # Reward for increasing score
        if action == Directions.STOP:
            reward -= 10  # Penalty for stopping
        currentPos = gameState.getAgentState(self.index).getPosition()
        nextPos = nextState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()
        if currentPos in foodList:
            reward += 5  # Reward for eating food
        enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and self.getMazeDistance(nextPos, a.getPosition()) < 2]
        if ghosts:  # Check if ghosts are too close
            reward -= 200  # Big penalty for getting too close to a ghost
        return reward


class QLearningDefensiveAgent(FoundationAgent):
    """
    Defensive Q-Learning agent.
    """
    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8, **kwargs):
        super().__init__(index, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.weights = util.Counter({
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'distanceToFood': -1,
            'isScared': -100,
            'forwardMovement': -5,
        })
    
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.foodToDefend = self.getFoodYouAreDefending(gameState).asList()

    def getDefensiveReward(self, gameState, action, nextState):
        reward = 0
        myPos = nextState.getAgentState(self.index).getPosition()
        invaders = [a for a in self.getOpponents(nextState) if nextState.getAgentState(a).isPacman and nextState.getAgentState(a).getPosition() != None]
        numInvadersNext = len(invaders)
        numInvadersNow = len([a for a in self.getOpponents(gameState) if gameState.getAgentState(a).isPacman and gameState.getAgentState(a).getPosition() != None])
        if numInvadersNow > numInvadersNext:
            reward += 100  # Reward for reducing invaders
        if action == Directions.STOP:
            reward -= 10  # Penalty for stopping
        if myPos == self.start:
            reward -= 5  # Penalty for being at start position (possibly got eaten)
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        if dists:
            reward += max(10 - min(dists), 0)  # Reward for being close to invaders (encouraging chase)
        return reward

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        
        features['onDefense'] = 1 if not myState.isPacman else 0
        invaders = [a for a in self.getOpponents(successor) if successor.getAgentState(a).isPacman and successor.getAgentState(a).getPosition() != None]
        features['numInvaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        else:
            features['invaderDistance'] = 0
        
        # Positioning towards the central defending area
        boundaryPosition = self.getBoundaryPosition(gameState)
        features['forwardMovement'] = self.getMazeDistance(myPos, boundaryPosition)

        return features

    def getBoundaryPosition(self, gameState):
        """
        Calculate a strategic position near the center of your side of the game board.
        This position is used for defending and intercepting invaders.
        """
        mapWidth = gameState.data.layout.width
        mapHeight = gameState.data.layout.height
        x = mapWidth // 2 if self.red else mapWidth // 2 - 1
        
        # Find the closest non-wall position to the center line
        for y in range(mapHeight // 2, mapHeight):
            if not gameState.hasWall(x, y):
                return (x, y)
        for y in range(mapHeight // 2, -1, -1):
            if not gameState.hasWall(x, y):
                return (x, y)
        return None


