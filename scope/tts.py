import random

#Train test split from scikit-learn
def tts(X, y, test_size=0.2, seed=None):
    '''
    Splits the data into training and testing sets
    
    Args:
        X : list of features
        y : list of labels
        test_size : data proportion for testing
        seed : same as random_state from scikit-learn
    
    Retuns:
        X_train, X_test, y_train, y_test
    '''


    if seed is not None:
        random.seed(seed)

    m = len(y)                  # total samples
    indexes = list(range(m))    # [0, 1, 2, ..., m-1]
    random.shuffle(indexes)

    test_count = int(m * test_size)

    test_indexes = indexes[:test_count]     #start from the first and stops at 20
    train_indexes = indexes[test_count:]    #starts from 20 and stops at the last value

    X_train = [X[i] for i in train_indexes]
    X_test = [X[i] for i in test_indexes]
    y_train = [y[i] for i in train_indexes]
    y_test = [y[i] for i in test_indexes]

    return X_train, X_test, y_train, y_test