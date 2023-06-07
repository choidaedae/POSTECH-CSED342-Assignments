#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    w = {'so': 1, 'touching': 1, 'quite': 0, 'impressive': 0, 'not': -1, 'boring': -1}  

    return w
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    templist = x.split(' ')
    # print(templist)
    ans = {}
    for tempword in templist:
        if (tempword in ans): ans[tempword] +=1
        else: ans[tempword] = 1

    return ans
        
        
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))
    
    # print(trainExamples)

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def predictor (x):
        if (dotProduct(weights, featureExtractor(x)) > 0): return 1
        else: return -1 

    def phi(x):
        return featureExtractor(x)

    for x, y in trainExamples: #weight initialize 
        for feature in featureExtractor(x):
            weights[feature] = 0

    def grad_nll(p, y): # compute the gradient of nll loss
        return -y * sigmoid(-y * dotProduct(p, weights))

    for i in range(numIters):
        for x, y in trainExamples:
            increment(weights, -eta * grad_nll(phi(x), y), phi(x))

    # print(weights)
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    wordlist = x.split(' ')
    phi = {}
    for i in range(0, len(wordlist)-(n-1)):
        tempkey = ""
        for j in range (i, i+n):
            tempkey += (wordlist[j] + " ")
        tempkey = tempkey[:-1]
        if (tempkey in phi): phi[tempkey] +=1
        else: phi[tempkey] = 1
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3, 'mu_y': 1.5}
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -1, 'mu_y': 0}, {'mu_x': 2, 'mu_y': 2}
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)

    def distance_sqr(i, j): # not exactly "distance", but sqaure of it 
        return examples_sqr[i] - 2 * dotProduct(examples[i], centroids[j]) + centroids_sqr[j]
        
    dim = len(examples[0]) 

    # print(examples)
    # print(K)
    
    centroids = random.sample(examples, K) # at first, set centroids with random sampling

    # to reduce overhead about dotproduct function call
    centroids_sqr = [dotProduct(c, c) for c in centroids] 
    examples_sqr = [dotProduct(p, p) for p in examples]
    
    # print(centroids)

    assignments = []
    for _ in range(maxIters):

        old_assignments = []

        # first, update z 
        for i in range(len(examples)):
            distances = {}
            for j in range(K):
                distances[j] = distance_sqr(i, j)
            temp = min(distances, key = distances.get)
            old_assignments.append(temp)
        
        if (old_assignments == assignments): 
            break # we need to terminate algorithm when it converges

        else: assignments = old_assignments

        # then, update u 
        means = [[{}, 0] for _ in range(K)]

        for i in range(len(examples)):
            center_idx = assignments[i]
            increment(means[center_idx][0], 1, examples[i]) # compute mean distance sum of each centroids
            means[center_idx][1] += 1

        for i, (mean, size) in enumerate(means):
            if size > 0:
                for dim, sum in list(mean.items()): # key is nth dimmension, and value is sum of distance of that dim
                    mean[dim] = sum / size
            centroids[i] = mean # update each entry of centroids value
            centroids_sqr[i] = dotProduct(mean, mean)
    
    total_cost = 0

    # finally, compute a total cost 
    for i in range(len(examples)): 
        center_idx = assignments[i]
        total_cost += distance_sqr(i, center_idx)

    return centroids, assignments, total_cost

    # END_YOUR_ANSWER

