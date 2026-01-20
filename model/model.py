import sys
import os

# Adiciona a pasta pai ao caminho de busca do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #For temporary testing

from processing.cleaning import clean_data
from processing.scaler import standardize_df as df_std
#########
# Model #
#########

data = clean_data()

def make_model(data):
    X_train = data.drop(['diagnosis'], axis=1) #Dropping the expected output and using the other values
    y_train = data['diagnosis'] #Using the Diagnosis as the expected output

    
    X_scaled = df_std(X_train)
    print(X_scaled.round(3))

make_model(data)