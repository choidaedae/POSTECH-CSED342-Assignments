'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, math, random, collections

# Class: ConditionalProb
class ConditionalProb:
    def __init__(self):
        self.t = 0
        self.condProb = {} # type is dictionary. it is referred to as Belief in the later problems.

    def setEnv(self, initial, transition, emission, states):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.states = states

############################################################
# Problem 1: Simplification
# setBinaryEnv():
# - delta: [δ] is the parameter governing the distribution
#   of the initial car's position
# - epsilon: [ε] is the parameter governing the conditional distribution
#   of the next car's position given the previos car's position
# - eta: [η] is the parameter governing the conditional distribution
#   of the sensor's measurement given the current car's position
# - states: possible values of $c_t$ and $d_t$
#   This corresponds to coordinates in later problems.
# - c_curr, c_prev and d_curr: $c_t$, $c_{t-1}$ and $d_t$, respectively.
# 
# normalize(): normalize self.CondProb to suit the probability distribution.
#
# observe():
# - d: observation $d_t$ at time [self.t]  
# - Update a conditional probability [self.condProb] of $c_t$ 
#   given observation $d_0, d_1, ..., d_t-1$ and $d_t$
#
# observeSeq(): 
# - [d_list]: $d_s$, ..., $d_t$
# - Estimate a conditional probability [self.condProb] of $c_t$ 
#   given observation $d_0, d_1, ..., d_t$
#
# Notes:
# - initial, transition, emission, and normalize functions and
#   if statements in observe functions are just guides.
# - If you really want, you can remove the given functions.
############################################################

    def setBinaryEnv(self, delta, epsilon, eta):        
        states = set(range(2)) # 0,1 
    # BEGIN_YOUR_ANSWER (our solution is 22 lines of code, but don't worry if you deviate from this)
        def initial(c1): return delta if c1 == 0 else 1-delta  # remove this error before writing code
        def transition(c_curr, c_prev): return epsilon if c_prev != c_curr else 1-epsilon 
        def emission(d_curr, c_curr): return eta if d_curr != c_curr else 1-eta
        self.setEnv(initial, transition, emission, states)

    def normalize(self):
        cp_sum =  sum(self.getCondProb().values())
        # print(self.getCondProb().values())
        # print(cp_sum)
        for i, _ in enumerate(self.getCondProb()):
            self.condProb[i] = self.condProb[i] / cp_sum

    def observe(self, d):
        if self.t == 0:
            for c_0 in [0, 1]:
                # print(self.condProb)
                self.condProb[c_0] = self.emission(d, c_0) * self.initial(c_0)
        else:
            prevCondProb = [self.condProb[0], self.condProb[1]]
            for c_t in [0, 1]:
                sum = 0
                for c_prev in [0,1]:
                    sum += self.transition(c_t, c_prev) * prevCondProb[c_prev]
                self.condProb[c_t] = self.emission(d, c_t) * sum

        self.t = self.t + 1
        self.normalize()

   # END_YOUR_ANSWER

    def observeSeq(self, d_list):
        for d in d_list:
            self.observe(d)

    def getCondProb(self): return self.condProb



# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference:
    
    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = False ### ONLY USED BY GRADER.PY in case problem 3 has not been completed
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()
   
     
    ############################################################
    # Problem 2: 
    # Function: Observe (update the probablities based on an observation)
    # -----------------
    # Takes |self.belief| and updates it based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard 
    #                 deviation Const.SONAR_STD
    # 
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Don't forget to normalize self.belief!
    ############################################################

    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        cols = self.belief.getNumCols()
        rows = self.belief.getNumRows()
        for col in range(cols):
            for row in range(rows):
                xdiff = agentX - util.colToX(col)
                ydiff = agentY - util.rowToY(row)
                dist = math.sqrt(xdiff ** 2 + ydiff ** 2)
                probdens = util.pdf(dist, Const.SONAR_STD, observedDist)
                postProb = self.belief.getProb(row, col)
                self.belief.setProb(row, col, postProb * probdens)
        self.belief.normalize()
        # END_YOUR_ANSWER

    ############################################################
    # Problem 3: 
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb, which gives all
    #   ((oldTile, newTile), transProb) key-val pairs that you must consider.
    # - Other ((oldTile, newTile), transProb) pairs not in self.transProb have
    #   zero probabilities and do not need to be considered. 
    # - util.Belief is a class (constructor) that represents the belief for a single
    #   inference state of a single car (see util.py).
    # - Be sure to update beliefs in self.belief ONLY based on the current self.belief distribution. 
    #   Do NOT invoke any other updated belief values while modifying self.belief.
    # - Use addProb and getProb to manipulate beliefs to add/get probabilities from a belief (see util.py).
    # - Don't forget to normalize self.belief!
    ############################################################
    def elapseTime(self):
        if self.skipElapse: return ### ONLY FOR THE GRADER TO USE IN Problem 2
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        cols = self.belief.getNumCols()
        rows = self.belief.getNumRows()
        new_belief = util.Belief(rows, cols, 0)
        for (old, new), t in self.transProb.items():
            old_p = self.belief.getProb(old[0], old[1])
            new_belief.addProb(new[0], new[1], old_p * t)
        new_belief.normalize()
        self.belief = new_belief
        # END_YOUR_ANSWER
      
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief

        
# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter:
    
    NUM_PARTICLES = 200
    
    # Function: Init
    # --------------
    # Constructer that initializes an ParticleFilter object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in a dict of defaultdict
        # self.transProbDict[oldTile][newTile] = probability of transitioning from oldTile to newTile
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]
            
        # Initialize the particles randomly
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1
            
        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a defaultdict from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief
    
    ############################################################
    # Problem 4 (part a): 
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$. 
    # This algorithm takes two steps:
    # 1. Reweight the particles based on the observation.
    #    Concept: We had an old distribution of particles, we want to update these
    #             these particle distributions with the given observed distance by
    #             the emission probability. 
    #             Think of the particle distribution as the unnormalized posterior 
    #             probability where many tiles would have 0 probability.
    #             Tiles with 0 probabilities (no particles), we do not need to update. 
    #             This makes particle filtering runtime to be O(|particles|).
    #             In comparison, exact inference (problem 2 + 3), most tiles would
    #             would have non-zero probabilities (though can be very small). 
    # 2. Resample the particles.
    #    Concept: Now we have the reweighted (unnormalized) distribution, we can now 
    #             resample the particles and update where each particle should be at.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new particle.
    ############################################################
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
        oldParticles = self.particles
        newParticles = collections.defaultdict(int)

        for row, col in oldParticles:
            xdiff = (agentX - util.colToX(col))
            ydiff = (agentY - util.rowToY(row))
            dist = math.sqrt(xdiff ** 2 + ydiff ** 2)
            probdens = util.pdf(dist, Const.SONAR_STD, observedDist)
            oldParticles[(row, col)] *= probdens
        
        for i in range(self.NUM_PARTICLES):
            sample = util.weightedRandomChoice(oldParticles)
            newParticles[sample] += 1

        self.particles = newParticles
        # END_YOUR_ANSWER
        self.updateBelief()
    
    ############################################################
    # Problem 4 (part b): 
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Read |self.particles| (defaultdict) corresonding to time $t$ and writes
    # |self.particles| corresponding to time $t+1$.
    # This algorithm takes one step
    # 1. Proposal based on the particle distribution at current time $t$:
    #    Concept: We have particle distribution at current time $t$, we want to
    #             propose the particle distribution at time $t+1$. We would like
    #             to sample again to see where each particle would end up using
    #             the transition model.
    #
    # Notes:
    # - transition probabilities is now using |self.transProbDict|
    # - Use util.weightedRandomChoice() to sample a new particle.
    # - To pass the grader, you must loop over the particles using
    #       for tile in self.particles
    #   and call util.weightedRandomChoice() $once per particle$ on the tile.
    ############################################################
    def elapseTime(self):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        newParticles = collections.defaultdict(int)
        for particle, count in self.particles.items():
            if particle in self.transProbDict:
                for i in range(count):
                    weights = self.transProbDict[particle]
                    sample = util.weightedRandomChoice(weights)
                    newParticles[sample] += 1
        self.particles = newParticles
        # END_YOUR_ANSWER
        
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief
