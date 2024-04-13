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

        if self.index == 0:
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
        if self.index == 0:
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

        deadEndMoves = {
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

        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and not a.scaredTimer > 1 and a.getPosition()]
        # scaredGhosts = len([a for a in enemies if a.scaredTimer > 0]) > 0 

        print("CURRENT POSITION:", currPos)
        print("ACTION:", action)

        # Don't want to stop
        if action == Directions.STOP:
            features['stop'] = 1

        # Don't want to reverse
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        minGhostDistance = float('inf')
    
        ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 1 and a.getPosition() != None]
        if len(ghosts) > 0:
            minGhostDistance = min([self.getMazeDistance(successorPos, a.getPosition()) for a in ghosts]) + 1
            features['distanceToGhost'] = minGhostDistance

        successorActions = successor.getLegalActions(self.index)

        # Filter out the stop position from successorActions and also filter out any actions which would result in the agent moving back to the same position
        successorActions = [a for a in successorActions if a != 'Stop' and self.getSuccessor(successor, a).getAgentPosition(self.index) != currPos]

        # print("Successor Actions:", successorActions)

        flippedPos = currPos
        if self.red:
            flippedPos = (31-currPos[0], 15-currPos[1])
            if deadEndMoves.get(flippedPos) is not None and deadEndMoves.get(flippedPos).get('action') == self.flipDirection(action) and minGhostDistance <=  deadEndMoves.get(flippedPos).get('length'):
              features['deadEnd'] = 1
        else:
          if deadEndMoves.get(flippedPos) is not None and deadEndMoves.get(flippedPos).get('action') == action and minGhostDistance <=  deadEndMoves.get(flippedPos).get('length'):
              features['deadEnd'] = 1

        # Check if successor action is a dead end, which PacMan shuoldn't take if there's a ghost nearby
        if len(successorActions) == 0 and minGhostDistance <= 3:
          features['deadEnd'] = 1

        if features['deadEnd'] == 1:
          print("Moving", action ,"is a dead end from this position", currPos)

        # Return to home if 10 (arbitrary number for now) pellets are being held by the agent
        pelletsHeld = gameState.getAgentState(self.index).numCarrying
        
        print("PELLETS HELD:", pelletsHeld)

        # If my agent is a ghost
        if not gameState.getAgentState(self.index).isPacman:
            self.desiredPellets = (3 * len(foodList)) // 10

        print("DESIRED PELLETS:", self.desiredPellets)

        if pelletsHeld >= self.desiredPellets:
            features['distanceToHome'] = -self.getMazeDistance(successorPos, self.start)

        # Get distance to center line
        centerLine = 16 if self.red else 17
        distanceToCenter = abs(successorPos[0] - centerLine)

        if len(foodList) > 0:
            minFoodDistance = min([self.getMazeDistance(successorPos, food) for food in foodList])
            features['distanceToFood'] = minFoodDistance

        if pelletsHeld >= 1 and distanceToCenter <= 1 and features['distanceToFood'] > 2:
            features['distanceToHome'] = -self.getMazeDistance(successorPos, self.start)

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

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return { 'distanceToHome': 5, 'successorScore': 100,  'distanceToGhost': 30, 'distanceToFood': -1, 'deadEnd': -500, 'stop': -175, 'reverse': -2}
    
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

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}