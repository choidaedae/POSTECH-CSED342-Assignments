import collections
import math

############################################################
# Problem 1a
def denseVectorDotProduct(v1, v2):    
    """
    Given two dense vectors |v1| and |v2|, each represented as list,
    return their dot product.
    You might find it useful to use sum(), and zip() and a list comprehension.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return sum([a * b for (a, b) in zip(v1, v2)]) 
    # END_YOUR_ANSWER

############################################################
# Problem 1b
def incrementDenseVector(v1, scale, v2):
    """
    Given two dense vectors |v1| and |v2| and float scalar value scale, return v = v1 + scale * v2.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    v = []
    for i in range (len(v1)):
        v.append(v1[i] + scale * v2[i])
    return v
    # END_YOUR_ANSWER

############################################################
# Problem 1c
def dense2sparseVector(v):
    """
    Given a dense vector |v|, return its sparse vector form,
    represented as collection.defaultdict(float).
    
    For exapmle:
    >>> dv = [0, 0, 1, 0, 3]
    >>> dense2sparseVector(dv)
    # defaultdict(<class 'float'>, {2: 1, 4: 3})
    
    You might find it useful to use enumerate().
    """
    sv = collections.defaultdict(float)
    for key, value in enumerate(v):
        if value != 0:
            sv[key] = value
    return sv

############################################################
# Problem 1d
def sparseVectorDotProduct(v1, v2):  # -> sparse vector product, dense vectoer product, dense sparse matmul
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float),
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    ans = 0
    for key1, value1 in v1.items():
        for key2, value2 in v2.items():
            if key1== key2:
                ans += value1*value2
 
    return ans
    # END_YOUR_ANSWER

############################################################
# Problem 1e
def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, return v = v1 + scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    ans = {}
    for key1, value1 in v1.items():
        ans[key1] = value1
    for key2, value2 in v2.items():
        if key2 in ans:
            ans[key2] += scale * value2
        else: ans[key2] = scale * value2
    return ans
    # END_YOUR_ANSWER

############################################################
# Problem 2a
def minkowskiDistance(loc1, loc2, p = math.inf): 
    """
    Return the Minkowski distance for p between two locations,
    where the locations are n-dimensional tuples.
    the Minkowski distance is generalization of
    the Euclidean distance and the Manhattan distance. 
    In the limiting case of p -> infinity,
    the Chebyshev distance is obtained.
    
    For exapmle:
    >>> p = 1 # manhattan distance case
    >>> loc1 = (2, 4, 5)
    >>> loc2 = (-1, 3, 6)
    >>> minkowskiDistance(loc1, loc2, p)
    # 5

    >>> p = 2 # euclidean distance case
    >>> loc1 = (4, 4, 11)
    >>> loc2 = (1, -2, 5)
    >>> minkowskiDistance = (loc1, loc2)  # 9

    >>> p = math.inf # chebyshev distance case
    >>> loc1 = (1, 2, 3, 1)
    >>> loc2 = (10, -12, 12, 2)
    >>> minkowskiDistance = (loc1, loc2, math.inf)
    # 14
    
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)

    if (p == math.inf): # exception: p = math.inf -> Chebyshev distance
        dis = []
        for (a, b) in zip(loc1, loc2):
            dis.append(abs(a-b))
        return max(dis)
        
    ans = pow(sum([pow(abs(a-b),p) for (a, b) in zip(loc1, loc2)]), 1/p) # other case
    return ans
    # END_YOUR_ANSWER

############################################################
# Problem 2b
def getLongestWord(text):
    """
    Given a string |text|, return the longest word in |text|. 
    If there are ties, choose the word that comes first in the alphabet.
    
    For example:
    >>> text = "tiger cat dog horse panda"
    >>> getLongestWord(text) # 'horse'
    
    Note:
    - Assume there is no punctuation and no capital letters.
    
    Hint:
    - max/min function returns the maximum/minimum item with respect to the key argument.
    """

    # BEGIN_YOUR_ANSWER (our solution is 4 line of code, but don't worry if you deviate from this)
    temp = text.split(' ')
    max = -1
    for word in temp: 
        if(max < len(word)):
            max = len(word)
    wlist = []
    for word in temp: 
        if(max == len(word)):
            wlist.append(word)

    return (min(wlist))
        
    # END_YOUR_ANSWER

############################################################
# Problem 2c
def getFrequentWords(text, freq):
    """
    Splits the string |text| by whitespace
    and returns a set of words that appear at a given frequency |freq|.
    """
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    wlist = text.split(' ')
    dic = dict()
    for word in wlist:
        try: dic[word] = dic[word] + 1
        except: dic[word] = 1
    ans = set()
    for (k, v) in dic.items():
        if (v==freq):
            ans.add(k)
    return ans

    # END_YOUR_ANSWER 