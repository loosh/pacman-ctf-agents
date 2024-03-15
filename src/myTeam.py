from captureAgents import CaptureAgent
import random, util
from game import Directions
import numpy as np
from util import nearestPoint, manhattanDistance

##################
# Game Constants #
##################

TRAINING = True
WEIGHT_PATH = 'weights_MY_TEAM.json'

#################
# Team Creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningOffensiveAgent', second='QLearningDefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class FoundationAgent(CaptureAgent):
    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8):
        super().__init__(index)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.weights = util.Counter()
        self.lastPositions = []
        self.repeatedActions = {}

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.lastPositions = [self.start for _ in range(5)]

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return sum(self.weights[f] * features[f] for f in features)

    def update(self, gameState, action, nextState, reward):
        features = self.getFeatures(gameState, action)
        difference = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(gameState, action)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def getValue(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return 0.0
        values = [self.getQValue(gameState, a) for a in actions]
        return max(values)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        features['bias'] = 1.0
        return features

    def getSuccessor(self, gameState, action):
        return gameState.generateSuccessor(self.index, action)

class QLearningOffensiveAgent(FoundationAgent):
    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8, optimisticSampling=True):
        super().__init__(index, epsilon, alpha, gamma)
        self.weights = util.Counter({
            'distanceToFood': -10,  # Encourage moving towards food
            'distanceToGhost': 20,  # Discourage getting close to ghosts
            'stop': -100,
            'reverse': -20,
            'numCarrying': 5,
            'distanceToSafeZone': -1,
            'distanceFromStart': 20,
            'stuckPenalty': -400,
            'exploreBias': 300,
            'scaredDistanceToEnemy': 10,
            'returnToFriendlySide': 50,  # Encourage returning to friendly side when carrying pellets
        })
        self.optimisticSampling = optimisticSampling

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to the nearest ghost
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        if len(defenders) > 0:
            distances = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
            features['distanceToGhost'] = min(distances)

        features['numCarrying'] = myState.numCarrying
        if myState.numCarrying > 0:
            features['distanceToSafeZone'] = self.getMazeDistance(myPos, self.start)
            # Encourage returning to the friendly side when carrying pellets
            middleX = gameState.data.layout.width // 2
            if self.red:
                if myPos[0] <= middleX:
                    features['returnToFriendlySide'] = 1
            else:
                if myPos[0] >= middleX:
                    features['returnToFriendlySide'] = 1

        features['distanceFromStart'] = self.getMazeDistance(myPos, self.start)

        if self.lastPositions[-5:].count(myPos) == 5:
            features['stuckPenalty'] = 1

        if len(self.lastPositions) > 5 and self.lastPositions[-1] == self.lastPositions[-3] and self.lastPositions[-2] == self.lastPositions[-4]:
            features['exploreBias'] = 1

        if myState.scaredTimer > 0:
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() is not None]
            if enemyPacmen:
                minDistance = min(self.getMazeDistance(myPos, a.getPosition()) for a in enemyPacmen)
                features['scaredDistanceToEnemy'] = minDistance

        return features

class QLearningDefensiveAgent(FoundationAgent):
    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8, optimisticSampling=True):
        super().__init__(index, epsilon, alpha, gamma)
        self.weights = util.Counter({
            'onDefense': 100,
            'invadersDistance': -10,
            'stop': -100,
            'reverse': -20,
            'numInvaders': -10,
            'distanceFromStart': 20,
            'stuckPenalty': -200,
            'exploreBias': 200,
            'scaredDistanceToEnemy': 10,  # Added feature for scared state
        })
        self.optimisticSampling = optimisticSampling

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['onDefense'] = 1 if not myState.isPacman else 0
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders] if invaders else [0]
        features['invadersDistance'] = min(dists, default=0)

        middleX = gameState.data.layout.width // 2
        if self.red:
            features['avoidEnemySide'] = -100 if myPos[0] >= middleX else 1
        else:
            features['avoidEnemySide'] = -100 if myPos[0] <= middleX else 1

        features['distanceFromStart'] = self.getMazeDistance(myPos, self.start)

        if self.lastPositions[-5:].count(myPos) == 5:
            features['stuckPenalty'] = 1

        if len(self.lastPositions) > 5 and self.lastPositions[-1] == self.lastPositions[-3] and self.lastPositions[-2] == self.lastPositions[-4]:
            features['exploreBias'] = 1

        if myState.scaredTimer > 0:
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() is not None]
            if enemyPacmen:
                minDistance = min(self.getMazeDistance(myPos, a.getPosition()) for a in enemyPacmen)
                features['scaredDistanceToEnemy'] = minDistance

        return features