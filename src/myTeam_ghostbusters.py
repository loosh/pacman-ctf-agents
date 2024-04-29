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
from game import Actions
import game
from util import nearestPoint
import json

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = True
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
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.weights = util.Counter()

        # Load Q-values from weights file
        self.loadQValues(WEIGHT_PATH)

        print("Loaded weights:", self.weights)
        self.lastAction = None

        myTeam = self.getTeam(gameState)

        opponentIndices = self.getOpponents(gameState)
        if self.index == myTeam[0]:
            for i, opponentIndex in enumerate(opponentIndices):
                particleFilters[i].initializeParticles(gameState, opponentIndex, first=True)

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

        self.ghost_estimates = []

        myTeam = self.getTeam(gameState)
        opponentIndices = self.getOpponents(gameState)
        for i, opponentIndex in enumerate(opponentIndices):
            particleFilters[i].observe(gameState.getAgentDistances()[opponentIndex], gameState, opponentIndex, self.index)
            if self.index == myTeam[0]:
                particleFilters[i].elapseTime()
            positions = particleFilters[i].getBeliefDistribution()
            self.ghost_estimates.append((max(positions, key=positions.get), gameState.getAgentState(opponentIndex)))

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

class ParticleFilter(CaptureAgent):
    def __init__(self, numParticles):
        self.walls = None
        self.numParticles = numParticles

    def initializeParticles(self, gameState, ghostIndex, first=False):
        if not self.walls:
            self.walls = gameState.getWalls().asList()
        self.particles = []
        ghostStartPosition = gameState.getInitialAgentPosition(ghostIndex)
        legalPositions = [p for p in gameState.getWalls().asList(False)]
        for i in range(self.numParticles):
              self.particles.append(ghostStartPosition if first else legalPositions[i % len(legalPositions)])

    def observe(self, observation, gameState, ghostIndex, pacmanIndex):
        noisyDistance = observation
        pacmanPosition = gameState.getAgentPosition(pacmanIndex)
        ghostPosition = gameState.getAgentPosition(ghostIndex)
        allPossible = util.Counter()

        if ghostPosition is not None:
            self.particles = [ghostPosition for i in range(self.numParticles)]
            return

        for p in self.particles:
            trueDistance = util.manhattanDistance(p, pacmanPosition)

            allPossible[p] += gameState.getDistanceProb(trueDistance, noisyDistance)

        if allPossible.totalCount() == 0:
            self.initializeParticles(gameState, ghostIndex)
            return

        allPossible.normalize()
        self.particles = [util.sample(allPossible) for i in range(self.numParticles)]

    def elapseTime(self):
      newParticles = []

      for p in self.particles:
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        legalPositions = [Actions.getSuccessor(p,a) for a in directions if Actions.getSuccessor(p, a) not in self.walls]
        newParticles.append(random.choice(legalPositions))
        
      self.particles = newParticles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        beliefDistribution = util.Counter()
        for particle in self.particles:
            beliefDistribution[particle] += 1
        beliefDistribution.normalize()
        return beliefDistribution
        
particleFilters = [ParticleFilter(400) for i in range(2)]

class PelletChaserAgent(QLearningAgent):
    
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.maxDistance = 70
        self.ghostAvoidanceDistance = 10
        self.numPellets = len(self.getFood(gameState).asList())
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

        numCarrying = gameState.getAgentState(self.index).numCarrying
        successorActions = successor.getLegalActions(self.index)

        # Filter out the stop position from successorActions and also filter out any actions which would result in the agent moving back to the same position
        successorActions = [a for a in successorActions if a != 'Stop' and self.getSuccessor(successor, a).getAgentPosition(self.index) != currPos]
        ghosts = [pos for (pos, state) in self.ghost_estimates if not state.isPacman and not state.scaredTimer > 0]

        features['eatsFood'] = 0
        if len(successorFoodList) < len(foodList):
          features['eatsFood'] = 1

        scaredGhosts = [pos for (pos, state) in self.ghost_estimates if not state.isPacman and state.scaredTimer > 0]
        if len(scaredGhosts) > 0:
            features['hasPowerPellet'] = 1 if len(scaredGhosts) > 0 else 0

        if features['eatsFood'] == 1:
            features[f'eatsPelletAt{str(currPos)}'] = 1
        elif currPos in foodList:
            features[f'eatsPelletAt{str(currPos)}'] = 0

        currStartDist = self.getMazeDistance(currPos, self.start)
        successorStartDist = self.getMazeDistance(successorPos, self.start)

        capsulePositions = self.getCapsules(gameState)
        if len(capsulePositions) > 0:
            if successorPos in capsulePositions:
              features['eatsCapsule'] = 1
            features['distanceToCapsule'] = min([self.getMazeDistance(currPos, capsule) for capsule in capsulePositions]) / self.maxDistance

        if len(ghosts) > 0:
            minDistanceToGhost = min([self.getMazeDistance(currPos, ghost) for ghost in ghosts])
            successorMinDistanceToGhost = min([self.getMazeDistance(successorPos, ghost) for ghost in ghosts])
            features['minDistanceToGhost'] = minDistanceToGhost / self.maxDistance

            if minDistanceToGhost < 5 and successorMinDistanceToGhost < minDistanceToGhost:
                features['movesTowardsGhost'] = 1

        features['returnsFood'] = 0
        if gameState.getAgentState(self.index).isPacman:
          features[f'currPos{str(currPos)}'] = 1

        minDistanceToFood = min([self.getMazeDistance(currPos, food) for food in foodList]) 
        if numCarrying > 0 and not features['movesTowardsGhost'] == 1 and len(ghosts) > 0:
            features['returnsFood'] = 1 if successorStartDist < currStartDist else 0
        
        if minDistanceToFood == 1 and (len(ghosts) == 0 or minDistanceToGhost > 3):
            features['returnsFood'] = 0

        elif len(foodList) > 2 and not features['movesTowardsGhost'] == 1:
          remainingPellets = len(foodList)
          self.ghostAvoidanceDistance = max(1, 10 - (20 - remainingPellets) * 9 / 18)
          
          foodListCopy = foodList.copy()
          if len(ghosts) > 0:
            foodList = [food for food in foodList if min([self.getMazeDistance(food, ghost) for ghost in ghosts]) > self.ghostAvoidanceDistance]
          if len(foodList) == 0:
            foodList = foodListCopy
          
          minDistanceToFood = min([self.getMazeDistance(currPos, food) for food in foodList]) 
          closestFood = min(foodList, key=lambda food: self.getMazeDistance(currPos, food))
          successorMinDistanceToFood = min([self.getMazeDistance(successorPos, food) for food in foodList])

          features[f'movesTowardsFood{str(closestFood)}'] = 1

          if successorMinDistanceToFood < minDistanceToFood:
              features['movesCloserToFood'] = 1
          
          features['minDistanceToFood'] = minDistanceToFood / self.maxDistance

        if action == Directions.STOP:
            features['stops'] = 1

        # print("Current Position", currPos)
        # print("Action", action)
        # print(features)
        # print("=====================================")


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
      
        reward = 0

        if features['eatsFood'] == 1:
            reward += 1
        if features['movesCloserToFood'] == 1:
            reward += 0.1
        if features['returnsFood'] == 1:
            reward += 2
        if features['movesCloserToGhost'] == 1:
            reward -= 0.5
        if features['eatsCapsule'] == 1:
            reward += 0.5

        if nextPos == pos:
            reward -= 1

        # If PacMan dies
        if nextPos == self.start and state.getAgentState(self.index).numCarrying > 0:
            reward -= 0.6

        return reward 

    def flipDirection(self, direction):
        if direction == 'North':
            return 'South'
        if direction == 'South':
            return 'North'
        if direction == 'East':
            return 'West'
        if direction == 'West':
            return 'East' 

class DefensiveAgent(QLearningAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.centerPellet = (14,9) if self.red else (17,6)
        self.sections = [1, 5, 10, 14] if self.red else [14, 10, 5, 1]
        self.entryPoints = [(11,2), (12,6), (12, 13)] if self.red else [(20, 13), (19, 9), (19, 2)]

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        invaderPositions = [pos for (pos, state) in self.ghost_estimates if state.isPacman]

        # print(invaderPositions)

        currPos = gameState.getAgentPosition(self.index)
        successorPos = successor.getAgentPosition(self.index)
        successorState = successor.getAgentState(self.index)

        isScared = gameState.getAgentState(self.index).scaredTimer > 0

        if len(invaderPositions) > 0:
            # Get closest invader's position
            closestInvaderPos = min(invaderPositions, key=lambda x: self.getMazeDistance(currPos, x))

            currentDistToInvader = self.getMazeDistance(currPos, closestInvaderPos)
            if currentDistToInvader <= 3:
              successorDistToInvader = self.getMazeDistance(successorPos, closestInvaderPos)

              if successorDistToInvader < currentDistToInvader:
                  features['movingTowardsInvader'] = 1
              elif isScared:
                  features['movingTowardsInvader'] = 1
            elif not isScared:
              y_cord = closestInvaderPos[1]
              section = 0
              for i in range(len(self.sections) - 1):
                  if self.red:
                    if y_cord >= self.sections[i] and y_cord <= self.sections[i+1]:
                      section = i
                      break
                  else:
                    if y_cord <= self.sections[i] and y_cord >= self.sections[i+1]:
                      section = i
                      break
              entryPoint = self.entryPoints[section]
              distToEntryPoint = self.getMazeDistance(currPos, entryPoint)
              successorDistToEntryPoint = self.getMazeDistance(successorPos, entryPoint)
              if successorDistToEntryPoint < distToEntryPoint:
                  features['movingToEntryPoint'] = 1
        elif not isScared:
          teamFoodList = self.getFoodYouAreDefending(gameState).asList()

          # Move to center pellet
          features['closerToFood'] = 0
          if self.centerPellet in teamFoodList:
              shiftedOne = (self.centerPellet[0], self.centerPellet[1] + (-1 if self.red else 1))
              distance = self.getMazeDistance(currPos, shiftedOne)
              nextDistance = self.getMazeDistance(successorPos, shiftedOne)
              if nextDistance < distance:
                  features['closerToFood'] = 1

        # On defense or offense
        features['onDefense'] = 1
        if successorState.isPacman:
            features['onDefense'] = 0
           
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
      
        reward = 0

        nextEnemies = [nextState.getAgentState(i)
                    for i in self.getOpponents(nextState)]
        nextInvaders = [a for a in nextEnemies if a.isPacman and a.getPosition()]

        if features['movingTowardsInvader'] == 1:
           reward += 0.2

        if features['movingToEntryPoint'] == 1:
            reward += 0.2

        if features['closerToFood'] == 1:
            reward += 0.2

        nextAgentState = nextState.getAgentState(self.index)

        if nextAgentState.isPacman:
            reward -= 5

        if nextPos == self.start:
            reward -= 0.8

        return reward