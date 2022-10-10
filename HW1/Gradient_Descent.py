import numpy as np
import matplotlib.pyplot as plt

def Loss(a,b):
    l = b - np.dot(np.transpose(a), a)
    l = np.multiply(l, l)
    l = np.sum(l)
    return l

if __name__ == '__main__':
    D = 10
    k = 10
    alpha = 0.001

    # Generate the A_ randomly with each element selected as a standard normal random variable
    A_ = np.reshape(np.random.normal(0, 1, D*k), (k, D))

    # Using A_ to build the B
    B = np.dot(np.transpose(A_), A_)

    # Generating A randomly
    A = np.reshape(np.random.normal(0, 1, D*k), (k, D))

    # Calculate the Loss L
    L = Loss(A, B)
    print(L)

    # Build a list to log the Loss
    history = []

    # Gradient descent
    while (L > 0.1 ** 15):
        G = 4 * np.dot(A, np.dot(np.transpose(A), A) - B)
        A = A - alpha * G
        L = Loss(A, B)
        history.append(L)

    plt.plot(history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(history[:100])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
