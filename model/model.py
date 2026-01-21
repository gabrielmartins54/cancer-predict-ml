from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def x_data(data):
    X = data.drop(['diagnosis'], axis=1) #Dropping the expected output and using the other values
    return X

def y_data(data):
    y = data['diagnosis'] #Using the Diagnosis as the expected output
    return y

def split(x, y, state=42, size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=size, random_state=state
    )
    return x_train, x_test, y_train, y_test

def train_test(x_train, x_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_hat = model.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, y_hat))
    print('Classification: \n', classification_report(y_test, y_hat))
    return y_hat

