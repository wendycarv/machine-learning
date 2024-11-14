from numpy import array, zeros

# Data: kINSP and kSEP
# Two datasets with three columns representing (x_1, x_2, y). 
# The first two values are the 2D features, and the third value
# is the label, where +1 and -1 indicate class membership.
# In these datasets:

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])

def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the weight vector.
    """
    # need to minimize 1/2|w|^2 to maximize the margin between classes
    w = zeros(len(x[0]))  # Initialize the weight vector
    
    '''
    Implement the logic to compute the weight vector.
    Hint: Loop over the data points and accumulate the contributions to the weight vector.
    '''  
    for i in range(len(x)):
        w += x[i]*y[i]*alpha[i]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Return the indices for all the support vectors.
    """
    # these define the margin of the classifier. they have the data values closest to the decision boundary
    # support vectors have a non-zero value (whereas non-support vectors have alpha = 0)
    support = set()
    
    '''
    Implement the logic to identify support vectors.
    Hints: what should each support vector satisfy? Use the tolerance value to handle precision issues.
    '''
    for i in range(len(x)):
        value = y[i] * (w.dot(x[i]) + b) # ideally 1 or close to it
        #print(f"Index {i}: y[i] = {y[i]} w.dot(x[i]) = {w.dot(x[i])}, b = {b} value = {value}, target = 1")

        # if the difference between the actual and expected values is less than the
        # tolerance, then it is quite close to the margin
        if abs(value - 1) <= tolerance:
        #    print(f"Support vector found at index {i}")
            support.add(i)

    return support

def find_slack(x, y, w, b):
    """
    Return the indices for all the slack vectors.
    """
    
    slack = set()
    
    '''
    Implement the logic to identify slack vectors.
    Hint: Slack vectors violate the margin constraint.
    '''
    for i in range(len(x)):
        value = y[i] * (w.dot(x[i]) + b) # ideally 1 or close to it

        if value < 1:
            slack.add(i)

    return slack


#if __name__ == '__main__':
#    x = kINSP[:, :-1]
#    y = kINSP[:, -1]

#    alpha = array([0.5] * len(x))

#    w = weight_vector(x, y, alpha)
#    print(f"Weight vector: {w}")

#    print("Support vector indices: ", find_support(x, y, w, 0))
#    print("Slack vector indices: ", find_slack(x, y, w, 0))


