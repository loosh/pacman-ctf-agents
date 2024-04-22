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
import json

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = False
DEBUG = False

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
WEIGHT_PATH = 'weights_ghostbusters.json'
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPSILON = 0.1

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
        self.weights = util.Counter()

        # Load Q-values from weights file
        self.loadQValues(WEIGHT_PATH)

        print("Loaded weights:", self.weights)
        self.lastAction = None

        CaptureAgent.registerInitialState(self, gameState)

    def computeValueFromQValues(self, gameState):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      action = self.computeActionFromQValues(gameState)
      return self.getQValue(gameState, action) if action else 0.0

    def computeActionFromQValues(self, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_value = float('-inf')
        best_action = None
        actions = gameState.getLegalActions(self.index)

        if len(actions) == 0:
            return None
        
        for action in actions:
            q_value = self.getQValue(gameState, action)
            if self.index == 0 and DEBUG:
              print("Action:", action)
              print("Q-Value:", q_value)
            if q_value > max_value:
                max_value = q_value
                best_action = action
            if q_value == max_value and util.flipCoin(0.5):
                max_value = q_value
                best_action = action

        return best_action
    
    def scaleWeights(self):
        """
        Scales the weights between -1 and 1 inclusive
        """
        max_abs_weight = max(abs(w) for w in self.weights.values())
        if max_abs_weight != 0:
            for feature in self.weights:
                self.weights[feature] /= max_abs_weight
                self.weights[feature] = max(min(self.weights[feature], 1), -1)
       
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        legalActions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        if TRAINING and self.lastAction:
            prevState = self.getPreviousObservation()
            reward = self.getReward(prevState, self.lastAction, gameState)
            self.update(prevState, self.lastAction, gameState, reward)
            if self.index == 0 and DEBUG:
              print("--------------------")
              print("Reward for action", self.lastAction, ":", reward)
        
        if len(legalActions) == 0:
            return None

        if TRAINING and util.flipCoin(EPSILON):
            if self.index == 0 and DEBUG:
              print("Taking random action")
            action = random.choice(legalActions)
            if self.index == 0 and DEBUG:
              print("Random action:", action)
        else:
            if self.index == 0 and DEBUG:
              print("Computing action from Q-values")
            action = self.computeActionFromQValues(gameState)
            if self.index == 0 and DEBUG:
              print("Action from Q-values:", action)

        self.lastAction = action
        return action

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
    
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0
        if self.index == 0 and DEBUG:
          print("Weights:", self.weights)
          print("Features:", self.getFeatures(state, action))
        for feature, value in self.getFeatures(state, action).items():
            total += value * self.weights.get(feature, 0)
        return total

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        if not TRAINING:
            return

        if self.weights != {}:
          self.scaleWeights()

        difference = (reward + DISCOUNT * self.getValue(nextState)) - self.getQValue(state, action)
        for feature, value in self.getFeatures(state, action).items():
              self.weights[feature] = self.weights[feature] + LEARNING_RATE * difference * value if feature in self.weights else 0

    def saveQValues(self, file_path):
        # Save Q-values to a file
        existing_weights = {}
        try:
            with open(file_path, 'r') as f:
                existing_weights = json.load(f)
        except FileNotFoundError:
            pass

        # Update weights for the current agent
        existing_weights[self.__class__.__name__] = self.weights

        # Save weights to the file
        with open(file_path, 'w') as f:
            json.dump(existing_weights, f)

    def loadQValues(self, file_path):
        # Load Q-values from a file
        with open(file_path, 'r') as f:
            all_weights = json.load(f)   
            valid_weights = all_weights and all_weights.get(self.__class__.__name__, None)
            self.weights = all_weights.get(self.__class__.__name__, util.Counter())

            if valid_weights and valid_weights != {}:
                self.scaleWeights()
                        
    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        super().final(self)

        if TRAINING:
          print("Final called")
          print("Weights:", self.weights)

          # Save the weights
          self.saveQValues(WEIGHT_PATH)


class PelletChaserAgent(QLearningAgent):
    
    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        
        currPos = gameState.getAgentPosition(self.index)
        successorPos = successor.getAgentPosition(self.index)

        foodList = self.getFood(gameState).asList()
        successorFoodList = self.getFood(successor).asList()

        enemies = self.getNearbyEnemies(gameState)
        numCarrying = gameState.getAgentState(self.index).numCarrying

        features['movingTowardsFood'] = 0
        features['movingTowardsGhost'] = 0
        if len(enemies) > 0 and numCarrying > 0:
            minGhostDist = min([self.getMazeDistance(currPos, ghost.getPosition()) for ghost in enemies])
            successorGhostDist = min([self.getMazeDistance(successorPos, ghost.getPosition()) for ghost in enemies])
            if minGhostDist > successorGhostDist:
                features['movingTowardsGhost'] = 1
        elif len(foodList) > 0:  
          minFoodDist = min([self.getMazeDistance(currPos, food) for food in foodList])
          successorFoodDist = min([self.getMazeDistance(successorPos, food) for food in foodList])
          if minFoodDist < successorFoodDist:
              features['movingTowardsFood'] = 1 
        
        features['eatsFood'] = 0
        if len(successorFoodList) < len(foodList):
            features['eatsFood'] = 1

        homeDist = self.getMazeDistance(currPos, self.start)
        successorHomeDist = self.getMazeDistance(successorPos, self.start)

        features['distanceToHome'] = 1 if homeDist > successorHomeDist and numCarrying > 2 else 0

        # Don't want to stop
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        # # Don't want to reverse
        # rev = Directions.REVERSE[gameState.getAgentState(
        #     self.index).configuration.direction]
        # if action == rev:
        #     features['reverse'] = 1
        # else:
        #     features['reverse'] = 0

        return features  
    
        self.deadEndMoves = {
            (8,13): {
                'action': 'South',
                'length': 8 + 1
            },
            (4, 10): {
                'action': 'West',
                'length': 38 + 1
            },
            (6, 10): { 
                'action': 'South',
                'length': 8 + 1
            },
            (6, 13): {
                'action': 'North',
                'length': 4 + 1
            },
            (7,2): {
                'action': 'North',
                'length': 4 + 1
            },
            (4,6): {
                'action': 'West',
                'length': 12 + 1
            },
            (12,9): {
                'action': 'West',
                'length': 4 + 1
            }
        }

    def flipDirection(self, direction):
        if direction == 'North':
            return 'South'
        if direction == 'South':
            return 'North'
        if direction == 'East':
            return 'West'
        if direction == 'West':
            return 'East' 

    def getReward(self, state, action, nextState):
        """
        Returns a reward for the state
        """
        pos = state.getAgentPosition(self.index)
        nextPos = nextState.getAgentPosition(self.index)

        if state is None:
            return 0
      
        reward = 0

        # If minDistance to food is less than previous state, reward
        foodList = self.getFood(state).asList()
        nextFoodList = self.getFood(nextState).asList()

        if pos == nextPos:
            reward -= 0.3

        if pos == self.start:
            reward -= 0.7

        enemies = self.getNearbyEnemies(state)
        nextEnemies = self.getNearbyEnemies(nextState)

        if len(nextEnemies) > len(enemies):
            reward -= 0.4
        elif len(enemies) > 0 and len(nextEnemies) > 0:
            minGhostDist = min([self.getMazeDistance(pos, ghost.getPosition()) for ghost in nextEnemies])
            nextMinGhostDist = min([self.getMazeDistance(nextPos, ghost.getPosition()) for ghost in nextEnemies])
            if nextMinGhostDist < minGhostDist:
                reward -= 0.6

        # Check if pacman moves closer to the nearest food
        if len(nextFoodList) > 0 and len(foodList) > 0:
            minFoodDist = min([self.getMazeDistance(pos, food) for food in foodList])
            nextMinFoodDist = min([self.getMazeDistance(nextPos, food) for food in nextFoodList])

            if nextMinFoodDist < minFoodDist:
                reward += 0.4
            else:
                reward -= 0.4

        if abs(self.getScore(nextState)) > abs(self.getScore(state)):
          if self.index == 0 and DEBUG:
            print("Score increased")
          reward += 1
          
        if len(nextFoodList) < len(foodList):
            if self.index == 0 and DEBUG:
              print("Eating food")
            reward += 0.5

        homeDist = self.getMazeDistance(pos, self.start)
        nextHomeDist = self.getMazeDistance(nextPos, self.start)
        numCarrying = state.getAgentState(self.index).numCarrying
        if numCarrying > 2:
            if homeDist > nextHomeDist:
                reward += 0.5

        # Pacman dies
        distanceFromStart = self.getMazeDistance(pos, self.start)
        if nextPos == self.start and distanceFromStart > 5:
            if self.index == 0 and DEBUG:
              print("Pacman died")
            reward -= 1

        return reward 
    
        # if not myState.isPacman:
        #     reward -= 10

        # Get distance from current position to start position
        # distanceFromStart = self.getMazeDistance(pos, self.start)
        # nextDistanceFromStart = self.getMazeDistance(nextPos, self.start)

        # myState = nextState.getAgentState(self.index)

    def getNearbyEnemies(self, state):
        enemies = [state.getAgentState(i)
                   for i in self.getOpponents(state)]
        ghosts = [a for a in enemies if not a.isPacman and not a.scaredTimer > 0 and a.getPosition()]
        return ghosts
  

class DefensiveAgent(QLearningAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index):
        self.recentlyVisitedFood = []
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        currPos = gameState.getAgentPosition(self.index)
        successorPos = successor.getAgentPosition(self.index)
        successorState = successor.getAgentState(self.index)

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if successorState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        currentEnemies = [gameState.getAgentState(i)
                    for i in self.getOpponents(gameState)]
        currentInvaders = [a for a in currentEnemies if a.isPacman and a.getPosition()]

        successorEnemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        successorInvaders = [a for a in successorEnemies if a.isPacman and a.getPosition()
                    != None]
        
        features['invadersExist'] = 1 if len(successorInvaders) > 0 else 0
        
        features['closerToInvaders'] = 0
        if len(successorInvaders) > 0 and len(currentInvaders) > 0:
            dists = min([self.getMazeDistance(
                currPos, a.getPosition()) for a in currentInvaders])
            successorDists = min([self.getMazeDistance(
                successorPos, a.getPosition()) for a in currentInvaders])

            if successorDists < dists:
                features['closerToInvaders'] = 1

        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0
        
        # rev = Directions.REVERSE[gameState.getAgentState(
        #     self.index).configuration.direction]
        # if action == rev:
        #     features['reverse'] = 1
        # else:
        #     features['reverse'] = 0

        teamFoodList = self.getFoodYouAreDefending(gameState).asList()

        # Intersection to remove food from visited that other pacman might have eaten
        self.recentlyVisitedFood = list(set(self.recentlyVisitedFood)&set(teamFoodList))

        # print("Recently visited food:", self.recentlyVisitedFood)
        # print("Team food list:", teamFoodList)
        # print("My position:", currPos)

        # print("Invaders:", successorInvaders)

        if currPos in teamFoodList:
            self.recentlyVisitedFood.append(currPos)

        if len(teamFoodList) == len(self.recentlyVisitedFood):
            self.recentlyVisitedFood = [currPos]

        teamFoodList = list(set(teamFoodList)-set(self.recentlyVisitedFood))

        features['closerToFood'] = 0
        if len(teamFoodList) > 0 and len(successorInvaders) == 0:
            minFoodDist = min([self.getMazeDistance(currPos, food) for food in teamFoodList])
            successorMinFoodDist = min([self.getMazeDistance(successorPos, food) for food in teamFoodList])

            distance = minFoodDist
            nextDistance = successorMinFoodDist

            if nextDistance < distance:
                features['closerToFood'] = 1
           
        return features

    def getReward(self, state, action, nextState):
        """
        Returns a reward for the state
        """
        pos = state.getAgentPosition(self.index)
        nextPos = nextState.getAgentPosition(self.index)

        features = self.getFeatures(state, action)

        if state is None:
            return 0
      
        reward = -0.01

        currentEnemies = [state.getAgentState(i)
                    for i in self.getOpponents(state)]
        currentInvaders = [a for a in currentEnemies if a.isPacman and a.getPosition()]

        nextEnemies = [nextState.getAgentState(i)
                    for i in self.getOpponents(nextState)]
        nextInvaders = [a for a in nextEnemies if a.isPacman and a.getPosition()]

        if len(nextInvaders) == 0 and features['closerToFood'] == 1:
            reward += 0.5

        nextAgentState = nextState.getAgentState(self.index)

        if nextAgentState.isPacman:
            reward -= 5

        if pos == nextPos:
            reward -= 0.7
        
        if len(currentInvaders) > 0 and len(nextInvaders) > 0:
          minDistanceToInvader = min([self.getMazeDistance(pos, a.getPosition()) for a in currentInvaders])
          nextMinDistanceToInvader = min([self.getMazeDistance(nextPos, a.getPosition()) for a in nextInvaders])
          if nextMinDistanceToInvader < minDistanceToInvader:
              reward += 0.7

        if nextPos == self.start:
            reward -= 0.8

        return reward