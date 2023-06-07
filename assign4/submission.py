import util, math, random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)

        total, next, deck = state # like (1, 1, (0, 1, 1))
        results = [] #results is tuple of (newstate, prob, reward) 

        if deck is None: return []

        sumCounts = sum(deck)

        if action == 'Take':

            if next is not None: #if card is peeked (we know that next card)
                
                newTotal = total + self.cardValues[next]

                if sumCounts > 1: newCounts = deck[:next] + (deck[next] - 1,) + deck[next+1:]
                else: newCounts = None

                if newCounts is None: reward = newTotal
                else: reward = 0

                results = [((newTotal, None, newCounts if newTotal <= self.threshold else None), 1, reward if newTotal <= self.threshold else 0)]

            else: #or not peeked 

                for i, count in enumerate(deck):

                    if count > 0: # if that card is in deck 
                        newTotal = total + self.cardValues[i]

                        if sumCounts > 1: newCounts = deck[:i] + (count - 1,) + deck[i+1:]
                        else: newCounts = None

                        if newCounts is None: reward = newTotal
                        else: reward = 0

                        prob = count / sumCounts

                        results.append(((newTotal, None, newCounts if newTotal <= self.threshold else None), prob, reward if newTotal <= self.threshold else 0))
                        
                        
        elif action == 'Peek':
            if next is not None: return []
            else: results = [((total, i, deck), count / sumCounts, -self.peekCost) for i, count in enumerate(deck) if count > 0]
        
        elif action == 'Quit':
            results = [((total, None, None), 1, total)] #reward = total 
        
        return results
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        Q_opt = self.getQ(state, action)
        lr = self.getStepSize()
        if isLast(newState): V_opt = 0
        else: V_opt = max(self.getQ(newState, action) for action in self.actions(newState))
        

        for feature, v in self.featureExtractor(state, action):
            self.weights[feature] = self.weights[feature] - lr * (Q_opt - (reward + self.discount * V_opt)) * v
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2] # s, a, r, s', a' -> SARSA 
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        Q_pi = self.getQ(state, action)
        Q_pi_next = self.getQ(newState, newAction)
        lr = self.getStepSize()

        for feature, v in self.featureExtractor(state, action):
            self.weights[feature] = self.weights[feature] - lr * (Q_pi - (reward + self.discount * Q_pi_next)) * v
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    feature = []
    f_name = (total, action)
    f_value = 1 # feature value is 1 
    feature.append((f_name, f_value))
    if counts is not None:
        second_f = list(counts)
        for i, count in enumerate(counts):
            if count > 0: second_f[i] = 1
        feature.append(((tuple(second_f), action), f_value))
        for i, count in enumerate(counts):
            feature.append(((i, count, action), f_value))

    return feature    
        

    # END_YOUR_ANSWER
