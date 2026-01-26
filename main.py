from model.model import *
from processing.cleaning import clean_data
from sklearn.preprocessing import StandardScaler
import pickle

def main():
    X = x_data(clean_data())
    y = y_data(clean_data())

    x_train, x_test, y_train, y_test = split(X, y)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = train_test(x_train, x_test, y_train, y_test)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()