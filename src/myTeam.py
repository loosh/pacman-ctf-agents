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
            if self.index == 0 and not TRAINING:
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
        
        if len(legalActions) == 0:
            return None

        if TRAINING and util.flipCoin(EPSILON):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(gameState)

        self.lastAction = action
        return action
    
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
    
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0
        if self.index == 0 and not TRAINING:
          print("Weights:", self.weights)
          print("Features:", self.getFeatures(state, action))
        for feature, value in self.getFeatures(state, action).items():
            total += value * self.weights[feature]
        return total

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        if not TRAINING:
            return

        self.scaleWeights()

        difference = (reward + DISCOUNT * self.getValue(nextState)) - self.getQValue(state, action)
        for feature, value in self.getFeatures(state, action).items():
            self.weights[feature] = self.weights[feature] + LEARNING_RATE * difference * value

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

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        
        currPos = gameState.getAgentPosition(self.index)
        successorPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()

        currentEnemies = [gameState.getAgentState(i)
                    for i in self.getOpponents(gameState)]
        currentGhosts = [a for a in currentEnemies if not a.isPacman and not a.scaredTimer > 1 and a.getPosition()]

        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and not a.scaredTimer > 1 and a.getPosition()]
        # scaredGhosts = len([a for a in enemies if a.scaredTimer > 0]) > 0 

        # print("CURRENT POSITION:", currPos)
        # print("ACTION:", action)

        # Don't want to stop
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        # Don't want to reverse
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        minGhostDistance = float('inf')
        currentGhostDistance = float('inf')
    
        # ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 1 and a.getPosition()]
        # currentGhosts = [a for a in currentEnemies if not a.isPacman and not a.scaredTimer > 1 and a.getPosition()]
        # features['minDistanceToGhost_3'] = 0
        # if len(ghosts) > 0:
        #     minGhostDistance = min([self.getMazeDistance(successorPos, a.getPosition()) for a in ghosts]) + 1
        #     # features['minDistanceToGhost_1'] = 1 if minGhostDistance <= 1 else 0
        #     # features['minDistanceToGhost_2'] = 1 if minGhostDistance <= 2 else 0
        #     if minGhostDistance:
        #         features['minDistanceToGhost_3'] = 1
        # if len(currentGhosts) > 0:
        #     currentGhostDistance = min([self.getMazeDistance(currPos, a.getPosition()) for a in currentGhosts]) + 1

        # if minGhostDistance < currentGhostDistance:
        #     features['closerToGhost'] = 1
        # else:
        #     features['closerToGhost'] = 0

        successorActions = successor.getLegalActions(self.index)

        # Filter out the stop position from successorActions and also filter out any actions which would result in the agent moving back to the same position
        successorActions = [a for a in successorActions if a != 'Stop' and self.getSuccessor(successor, a).getAgentPosition(self.index) != currPos]

        # print("Successor Actions:", successorActions)

        # features['deadEnd'] = 0

        # flippedPos = currPos
        # if self.red:
        #     flippedPos = (31-currPos[0], 15-currPos[1])
        #     if deadEndMoves.get(flippedPos) is not None and deadEndMoves.get(flippedPos).get('action') == self.flipDirection(action) and minGhostDistance <=  deadEndMoves.get(flippedPos).get('length'):
        #       features['deadEnd'] = 1
        # else:
        #   if deadEndMoves.get(flippedPos) is not None and deadEndMoves.get(flippedPos).get('action') == action and minGhostDistance <=  deadEndMoves.get(flippedPos).get('length'):
        #       features['deadEnd'] = 1

        # # Check if successor action is a dead end, which PacMan shuoldn't take if there's a ghost nearby
        # if len(successorActions) == 0 and minGhostDistance <= 3:
        #   features['deadEnd'] = 1

        # if features['deadEnd'] == 1:
          # print("Moving", action ,"is a dead end from this position", currPos)

        # Return to home if 10 (arbitrary number for now) pellets are being held by the agent
        pelletsHeld = gameState.getAgentState(self.index).numCarrying
        
        # If my agent is a ghost
        if not gameState.getAgentState(self.index).isPacman:
            self.desiredPellets = (3 * len(foodList)) // 10
# 
        # print("DESIRED PELLETS:", self.desiredPellets)

        # Get distance to center line
        centerLine = 16 if self.red else 17
        distanceToCenter = abs(successorPos[0] - centerLine)

        successorFoodList = self.getFood(successor).asList()
        if len(foodList) > 2:
            minFoodDistance = min([self.getMazeDistance(currPos, food) for food in foodList])
            minSuccessorFoodDistance = min([self.getMazeDistance(successorPos, food) for food in successorFoodList])

            # print("MIN FOOD DISTANCE:", minFoodDistance)
            # print("MIN SUCCESSOR FOOD DISTANCE:", minSuccessorFoodDistance)

            features['distanceToFood'] = 1 if minSuccessorFoodDistance < minFoodDistance else 0
        else:
            features['distanceToFood'] = 0

        # print(features['distanceToFood'])
        
        firstFoodPosition = (10, 14)
        redFirstFoodPosition = (21, 1)

        # if firstFoodPosition in foodList or redFirstFoodPosition in foodList:
        #     features['distanceToFood'] = 1

        # if pelletsHeld >= self.desiredPellets and features['distanceToFood'] > 5:
        #     features['headHome'] = 1

        # if pelletsHeld >= 1 and distanceToCenter <= 1 and features['distanceToFood'] > 2:
        #     features['headHome'] = 1

        successorDistanceToStart = self.getMazeDistance(successorPos, self.start)
        currentDistanceToStart = self.getMazeDistance(currPos, self.start)

        if len(foodList) <= 2:
            features['headHome'] = 1 if successorDistanceToStart < currentDistanceToStart else 0
        else:
            features['headHome'] = 0

        return features  

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
        currentPos = state.getAgentPosition(self.index)
        nextPos = nextState.getAgentPosition(self.index)

        if state is None:
            return 0
      
        reward = -0.001

        # If minDistance to food is less than previous state, reward
        currentFoodList = self.getFood(state).asList()
        nextFoodList = self.getFood(nextState).asList()

        # Get distance from current position to start position
        distanceFromStart = self.getMazeDistance(currentPos, self.start)
        nextDistanceFromStart = self.getMazeDistance(nextPos, self.start)

        currentEnemies = self.getNearbyEnemies(state)
        nextEnemies = self.getNearbyEnemies(nextState)

        minGhostDistance = float('inf')
        nextMinGhostDistance = float('inf')
        
        if len(nextEnemies) > 0:
            if len(currentEnemies) == 0:
                reward -= 2
            else:
              minGhostDistance = min([self.getMazeDistance(currentPos, a.getPosition()) for a in currentEnemies]) + 1
              nextMinGhostDistance = min([self.getMazeDistance(nextPos, a.getPosition()) for a in nextEnemies]) + 1

              if nextMinGhostDistance < minGhostDistance:
                  reward -= 10

        # Check if agent is pacman
        if not nextState.getAgentState(self.index).isPacman and nextDistanceFromStart > distanceFromStart:
          reward += 1

        rev = Directions.REVERSE[state.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            reward -= 1
        
        if action == Directions.STOP:
            reward -= 2

        if len(currentFoodList) <= 15:
            # if next position is closer to start
            if nextDistanceFromStart < distanceFromStart:
                reward += 50

        # if len(nextState.getLegalActions(self.index)) == 1 and nextMinGhostDistance >= 1 and nextMinGhostDistance <= 3:
        #     reward -= 0.1

        if len(currentFoodList) > 0:
            currentMinFoodDistance = min([self.getMazeDistance(currentPos, food) for food in currentFoodList])
            nextMinFoodDistance = min([self.getMazeDistance(nextPos, food) for food in nextFoodList])

            if nextMinFoodDistance < currentMinFoodDistance:
                reward += 1

        if self.getScore(nextState) > self.getScore(state):
            reward += 15

        if len(nextFoodList) < len(currentFoodList):
            reward += 5

        return reward 

    def getNearbyEnemies(self, state):
        enemies = [state.getAgentState(i)
                   for i in self.getOpponents(state)]
        ghosts = [a for a in enemies if not a.isPacman and not a.scaredTimer > 0 and a.getPosition()]
        return ghosts

    # def getWeights(self, gameState, action):
    #     """
    #     Normally, weights do not depend on the gamestate.  They can be either
    #     a counter or a dictionary.
    #     """
    #     return { 'distanceToHome': 5, 'successorScore': 100,  'distanceToGhost': 20, 'distanceToFood': -1, 'deadEnd': -500, 'stop': -175, 'reverse': -2}
    
class DefensiveAgent(QLearningAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1

        # Computes distance to invaders we can see
        currentEnemies = [gameState.getAgentState(i)
                    for i in self.getOpponents(gameState)]
        currentInvaders = [a for a in currentEnemies if a.isPacman and a.getPosition()]

        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['invadersExist'] = 1 if len(invaders) > 0 else 0
        
        features['closerToInvaders'] = 0
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            currentDists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in currentInvaders]

            if min(dists) < min(currentDists):
                features['closerToInvaders'] = 1

        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0
        
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        # if (len(invaders)==0):
        #     features['goToCenter'] = 1
        # else:
        #     features['goToCenter'] = 0

        return features

    def getReward(self, state, action, nextState):
        """
        Returns a reward for the state
        """
        currentPos = state.getAgentPosition(self.index)
        nextPos = nextState.getAgentPosition(self.index)

        if state is None:
            return 0
      
        reward = -0.001

        # If minDistance to food is less than previous state, reward
        currentFoodList = self.getFood(state).asList()
        nextFoodList = self.getFood(nextState).asList()

        # Get distance from current position to start position
        distanceFromStart = self.getMazeDistance(currentPos, self.start)
        nextDistanceFromStart = self.getMazeDistance(nextPos, self.start)

        centerLine = 16 if self.red else 17
        nextPosDistToCenter = self.getMazeDistance(nextPos, (centerLine, 7))
        currentPosDistToCenter = self.getMazeDistance(currentPos, (centerLine, 7))

        if nextPos == self.start:
            reward -= 5

        if nextPosDistToCenter < currentPosDistToCenter:
            reward += 3

        # Check if agent is pacman
        if not nextState.getAgentState(self.index).isPacman and nextDistanceFromStart > distanceFromStart:
          reward += 1

        rev = Directions.REVERSE[state.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            reward -= 1
        
        if action == Directions.STOP:
            reward -= 2

        if len(nextState.getLegalActions(self.index)) == 1:
            reward -= 0.1

        if self.getScore(nextState) > self.getScore(state):
            reward += 15

        return reward 

    # def getWeights(self, gameState, action):
    #     return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}