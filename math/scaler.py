###################
# Standardization #
###################

#Mimics the StandardScaler function from scikit-learn

def standardize(x: list):
    '''
    Args
        x : list containing the values to be standardized
    Returns
        z : list of standardized values
    '''

    #mean
    mean = sum(x) / len(x)

    #squared differences
    squared_diffs = [(i - mean) ** 2 for i in x]

    #variance 
    variance = sum(squared_diffs) / len(x)

    #standard deviation
    std = variance ** 0.5

    #z-score
    z = [(i - mean) / std for i in x]

    return z
    




