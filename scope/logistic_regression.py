import math


# Sigmoid activation function
def sigmoid(z):
    """
    Applies the sigmoid function to a scalar value.

    Mathematical form:
        σ(z) = 1 / (1 + e^(-z))

    Args:
        z : linear combination of inputs (w · x + b)

    Returns:
        A value between 0 and 1 representing probability
    """
    return 1 / (1 + math.exp(-z))


# Dot (scalar) product between feature vector and weights
def dot_product(x, w):
    """
    Computes the dot product between a feature vector and weight vector.

    Mathematical form:
        z = x₁w₁ + x₂w₂ + ... + xₙwₙ

    Args:
        x : feature vector (one sample)
        w : weight vector

    Returns:
        Scalar value z
    """
    return sum(x_i * w_i for x_i, w_i in zip(x, w))


# Logistic Loss — computes mean cost over the dataset
def compute_cost(X, y, w, b):
    """
    Computes the mean logistic loss (cost function).

    Mathematical form:
        J(w, b) = (1/m) * Σ[ -y log(ŷ) - (1 - y) log(1 - ŷ) ]

    Args:
        X : list of feature vectors (training data)
        y : list of expected outputs (labels)
        w : weight vector
        b : bias

    Returns:
        Mean cost over all training samples
    """
    m = len(y)          # Number of training examples
    total_cost = 0     # Accumulates total loss

    for i in range(m):
        # Model prediction for sample i
        f_wb = sigmoid(dot_product(X[i], w) + b)

        # Logistic loss for a single sample
        total_cost += (
            -y[i] * math.log(f_wb)
            - (1 - y[i]) * math.log(1 - f_wb)
        )

    return total_cost / m   # Mean cost


# Computes gradients of the cost function
def compute_gradient(X, y, w, b):
    """
    Computes the gradients of the cost function with respect to
    weights and bias.

    Mathematical form:
        ∂J/∂wⱼ = (1/m) * Σ[(ŷᶦ - yᶦ) * xⱼᶦ]
        ∂J/∂b  = (1/m) * Σ(ŷᶦ - yᶦ)

    Args:
        X : list of feature vectors
        y : list of expected outputs
        w : weight vector
        b : bias

    Returns:
        dj_dw : gradient for each weight
        dj_db : gradient for bias
    """
    m = len(y)          # Number of training examples

    # Initialize gradients
    dj_dw = [0] * len(w)
    dj_db = 0

    for i in range(m):
        # Model prediction for sample i
        f_wb_i = sigmoid(dot_product(X[i], w) + b)

        # Prediction error (ŷ - y)
        error = f_wb_i - y[i]

        # Accumulate gradients for each weight
        for j in range(len(w)):
            dj_dw[j] += error * X[i][j]

        # Accumulate gradient for bias
        dj_db += error

    # Compute mean gradients
    dj_dw = [d / m for d in dj_dw]
    dj_db /= m

    return dj_dw, dj_db


# Gradient Descent optimization
def gradient_descent(X, y, alpha=0.01, num_iters=1000):
    """
    Optimizes weights and bias using Gradient Descent.

    Update rules:
        wⱼ := wⱼ - α * ∂J/∂wⱼ
        b  := b  - α * ∂J/∂b

    Args:
        X : list of feature vectors
        y : list of expected outputs
        alpha : learning rate
        num_iters : number of iterations

    Returns:
        w : trained weights
        b : trained bias
    """
    # Initialize parameters
    w = [0] * len(X[0])   # One weight per feature
    b = 0

    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        # Update weights
        for j in range(len(w)):
            w[j] -= alpha * dj_dw[j]

        # Update bias
        b -= alpha * dj_db

    return w, b


# Predict class labels
def predict(X, w, b):
    """
    Predicts class labels for input samples.

    Decision rule:
        ŷ = 1 if σ(w · x + b) ≥ 0.5
        ŷ = 0 otherwise

    Args:
        X : list of feature vectors
        w : trained weights
        b : trained bias

    Returns:
        List of predicted class labels (0 or 1)
    """
    predictions = []

    for x in X:
        prob = sigmoid(dot_product(x, w) + b)
        predictions.append(1 if prob >= 0.5 else 0)

    return predictions
