from model.model import *
from processing.cleaning import clean_data
from sklearn.preprocessing import StandardScaler
import pickle

def main():
    X = x_data(clean_data())
    y = y_data(clean_data())

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = split(X, y)

    prediction = train_test(x_train, x_test, y_train, y_test)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(prediction, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()