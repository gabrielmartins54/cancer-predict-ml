import pandas as pd


def clean_data():
    '''
    Returns
        data : -> pandas DataFrame
    '''
    #read the data
    data = pd.read_csv('data/cancer_data.csv')

    #cleaning the data'
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    #mapping possible diagnosis
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) #   malignant will be 1, benign will be 0

    return data

data = clean_data()

    