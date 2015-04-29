from __future__ import division
import numpy as np
import numpy.testing as npt
import scipy.stats as stats

# Test the log-likelihood of Poisson
# Does the exp(log-likelihood) sum to 1?
# Since Poisson is a discrete distribution, we can add up the values
# Prior: x ~ Pois(lambda): f(x) = ((lambda**x)/x!)*exp(-lambda), where x = ki, lambda = alphaN

def log_Pois(x,alphaN):
    return x*np.log(alphaN) - alphaN - np.log(np.math.factorial(x)) 

alphaN = 0.5
array = np.zeros((1,21))
count = 0

for i in range(21):
    array[0,i] = log_Pois(i,alphaN)
    count += np.exp(array[0,i])
    # print array[0,i]

# print count 
npt.assert_almost_equal(count,1)
print "The exp(log-likelihood) of Poisson sums to 1, so it passed the assertion test!"