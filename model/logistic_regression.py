import math


#Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

#Scalar product
def dot_product(x, w):
    return sum(x_i * w_i for x_i, w_i in zip(x, w))    

#Logistic loss
def compute_cost(X, y, w, b):
    m = len(y)
    total_cost = 0

    for i in range(m):
        z = dot_product(X[i], w) + b
        y_hat = sigmoid(z)

        total_cost += (
            -y[i] * math.log(y_hat) - (1 - y[i]) * math.log(1 - y_hat)
        )
    return total_cost / m

#def compute_gradient

#def gradient_descent

#def predict