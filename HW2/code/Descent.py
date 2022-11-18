import numpy as np
import random
from plot_functions import plot_exp, plot_error, plot_direction

def random_vector(dim = 10):
    c = np.reshape(np.random.normal(0, 1, dim), (dim, 1))
    return c

def random_Square_Matrix(dim = 10):
    A = np.reshape(np.random.normal(0, 1, dim * dim), (dim, dim))
    return A

def generate_orthogonal_vector(x):
    # get the shape of x so that it work for all dimensional x
    shape = int(x.shape[0])

    # random generte a vector
    orthagonal_x = np.reshape(np.random.randn(shape), (shape, 1))

    # set this vector to orthogonal to x by changing the last element
    c = np.dot(np.transpose(x)[:, :shape-1], orthagonal_x[:shape-1])
    last_element = -c / x[-1][0]
    orthagonal_x[-1][0] = last_element

    return orthagonal_x


def Descent(G, Q, c, x_original, mode, termination, id, optimal_x, alpha=0.01, beta=0.01):
    x = np.copy(x_original)
    # Build a list to log the Loss
    history = []
    history.append(x)
    iter = 0

    # Gradient descent
    while (np.abs(np.max(G)) > termination):
        iter += 1
        G = np.dot(Q, x) - c

        # different type of optimization
        if "normal" in mode and "Momentum" not in mode:
            x = x - alpha * G

        elif "Newton" in mode:
            alpha = np.dot(np.transpose(G), G) / np.dot(np.dot(np.transpose(G), Q), G)
            x = x - alpha * G

        elif "Momentum_normal" in mode:
            # if it's the first iteration, we have no momentum at this point
            if iter == 1:
                x = x - alpha * G

            else:
                x = x - alpha * G + beta * (x - history[-2])

        elif "Momentum_optimal" in mode:
            # if it's the first iteration, we have no momentum at this point
            if iter == 1:
                alpha = np.dot(np.transpose(G), G) / np.dot(np.dot(np.transpose(G), Q), G)
                x = x - alpha * G
            else:
                q = x - history[-2]
                alpha = (np.dot(np.transpose(G), G) * np.dot(np.transpose(q), np.dot(Q, q)) - np.dot(np.transpose(q), G) * np.dot(np.transpose(G), np.dot(Q, q))) \
                        / (np.dot(np.transpose(G), np.dot(Q, G)) * np.dot(np.transpose(q), np.dot(Q, q)) - np.dot(np.transpose(G), np.dot(Q, q)) * np.dot(np.transpose(G), np.dot(Q, q)))
                beta = (np.dot(np.transpose(G), G) * np.dot(np.transpose(G), np.dot(Q, q)) - np.dot(np.transpose(q), G) * np.dot(np.transpose(G), np.dot(Q, G))) \
                        / (np.dot(np.transpose(G), np.dot(Q, G)) * np.dot(np.transpose(q), np.dot(Q, q)) - np.dot(np.transpose(G), np.dot(Q, q)) * np.dot(np.transpose(G), np.dot(Q, q)))
                x = x - alpha * G + beta * q

        elif "Momentum_orthogonal" in mode:
            q = generate_orthogonal_vector(G)
            alpha = (np.dot(np.transpose(G), G) * np.dot(np.transpose(q), np.dot(Q, q)) - np.dot(np.transpose(q),
                                                                                                 G) * np.dot(
                np.transpose(G), np.dot(Q, q))) \
                    / (np.dot(np.transpose(G), np.dot(Q, G)) * np.dot(np.transpose(q), np.dot(Q, q)) - np.dot(
                np.transpose(G), np.dot(Q, q)) * np.dot(np.transpose(G), np.dot(Q, q)))
            beta = (np.dot(np.transpose(G), G) * np.dot(np.transpose(G), np.dot(Q, q)) - np.dot(np.transpose(q),
                                                                                                G) * np.dot(
                np.transpose(G), np.dot(Q, G))) \
                   / (np.dot(np.transpose(G), np.dot(Q, G)) * np.dot(np.transpose(q), np.dot(Q, q)) - np.dot(
                np.transpose(G), np.dot(Q, q)) * np.dot(np.transpose(G), np.dot(Q, q)))
            x = x - alpha * G + beta * q


        history.append(x)
        if random.random() < 0.01:
            print(np.max(G), iter, beta, alpha)

    #plots
    plot_error(history, optimal_x, mode + '_' + 'error' + id, mode='log')
    plot_exp(history, optimal_x, mode + '_' + 'exp' + id)
    plot_direction(history, optimal_x, mode + '_' + 'direct' + id)

    return x, history


