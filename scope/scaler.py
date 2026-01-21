###################
# Standardization #
###################

#Mimics the StandardScaler function from scikit-learn


#v1 function to understand standardization proccess
def standardize(x):
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
    

def standardize_df(df):
    '''
    Standardize the dataframe values

    Args:
        df: -> Pandas cleaned dataframe
    Retuns:
        df_std: -> Pandas standardized dataframe
    '''

    df_std = df.copy()

    for col in df.columns:
        mean = df[col].mean()   #Returns mean
        std = df[col].std()     #Returns standard deviation
        df_std[col] = (df[col] - mean) / std    #Finally aplly z-score

    return df_std 



