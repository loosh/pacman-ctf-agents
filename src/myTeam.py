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
from game import Directions
import game
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
              first = 'PelletChaserAgent', second = 'DefensiveAgent'):
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

class QLearningAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]

        if self.index == 1:
          print("Actions:", actions)
          print("Values:", values)

        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        foodLeft = len(self.getFood(gameState).asList())
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        return features

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

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        # if offensive agent
        if self.index == 1:
          print("=====================")
          # print position of agent
          print(gameState.getAgentPosition(self.index))
          print(features)
          print(features * weights)
          print("=====================")
        return features * weights

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}
    
class PelletChaserAgent(QLearningAgent):
    
    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        pelletPositions = self.getFood(gameState).asList()
        successorPosition = successor.getAgentPosition(self.index)

        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        ghostPositions = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition()]

        scaredGhosts = len([a for a in enemies if a.scaredTimer > 0]) > 0 

        successorActions = successor.getLegalActions(self.index)
        currPosition = gameState.getAgentPosition(self.index)

        # Filter out the stop position from successorActions and also filter out any actions which would result in the agent moving back to the same position
        successorActions = [a for a in successorActions if a != 'Stop' and successor.getAgentPosition(self.index) != currPosition]

        # print(successorActions)

        if action == Directions.STOP:
            features['stop'] = 1

        if len(successorActions) == 1:
          print("Deadend based on successor actions")
          if sum(self.getMazeDistance(currPosition, ghostPosition) <= 3 for ghostPosition in ghostPositions) > 0:
            features['deadend'] = 1
        # else:
        #   numOfDeadEnds = 0

        #   for potential_action in successorActions:
        #     nextSuccessor = self.getSuccessor(successor, potential_action)
        #     nextSuccessorActions = nextSuccessor.getLegalActions(self.index)
        #     currPosition = nextSuccessor.getAgentState(self.index).getPosition()
        #     nextSuccessorActions = [a for a in nextSuccessorActions if a != 'Stop' and self.getSuccessor(nextSuccessor, a).getAgentState(self.index).getPosition() != currPosition]
            
        #     if len(nextSuccessorActions) == 1:
        #       numOfDeadEnds += 1
          
        #   if numOfDeadEnds == len(successorActions):
        #     print("Deadend based on successor actions length")
        #     if sum(self.getMazeDistance(currPosition, ghostPosition) <= 2 for ghostPosition in ghostPositions) > 0:
        #       features['deadend'] = 1

        # print("DEADEND FEATURE", features['deadend'])

        pelletsHeld = gameState.getAgentState(self.index).numCarrying
        foodLeft = len(self.getFood(gameState).asList())
        if (pelletsHeld / (foodLeft + pelletsHeld)) * 100 > 35:
            print("GOING BACK HOME", self.index)
            features['distanceToHome'] = -self.getMazeDistance(successorPosition, self.start)

        if len(ghostPositions) > 0:
          features['ghosts-1-step-away'] = -sum(self.getMazeDistance(successorPosition, ghostPosition) <= 1 for ghostPosition in ghostPositions)

        # print('Ghosts 1 step away', features['ghosts-1-step-away'])

        nearestPellet = min(pelletPositions, key=lambda x: self.getMazeDistance(successorPosition, x))
        distanceToNearestPellet = self.getMazeDistance(successorPosition, nearestPellet)

        # print('Distance to nearest pellet', distanceToNearestPellet)

        if not features['ghosts-1-step-away'] and distanceToNearestPellet: 
          features['minDistanceToFood'] = -distanceToNearestPellet

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return { 'distanceToHome': 25, 'successorScore': 100,  'ghosts-1-step-away': 100, 'minDistanceToFood': 1, 'deadend': -200, 'stop': -100 }
    
class DefensiveAgent(QLearningAgent):
   def getFeatures(self, gameState, action):
      features = util.Counter()
      if action == Directions.STOP:
            features['stop'] = 1
      return features
   
   def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return { 'stop': 100 }
      