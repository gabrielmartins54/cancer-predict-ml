from cleaning import cleaning as cl
#########
# Model #
#########

data = cl.clean_data()

def make_model(data):
    X_train = data.drop(['diagnosis'], axis=1) #Dropping the expected output and using the other values
    y_train = data['diagnosis'] #Using the Diagnosis as the expected output

