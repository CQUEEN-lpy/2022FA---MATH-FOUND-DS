import numpy as np

count = 0

def diagonal_matrix(beta):
    l = []
    for i in range(beta.shape[0]):
        tmp = [0] * beta.shape[0]
        tmp[i] = beta[i]
        l.append(tmp)
    B = np.array(l)
    return B

def new_R(X, index, alpha, beta, R):
    tmp_x = X[:index]
    beta_x = beta[:index]
    B_x = diagonal_matrix(beta_x)
    ax = np.dot(np.transpose(tmp_x), np.dot(B_x, tmp_x))
    tmp_R = R - ax
    return tmp_R

def relaxing_optimal(X, index, alpha, beta, R):
    tmp_x = X[:index]

    # calculate B, relaxing beta and alpha
    tmp_alpha = alpha[index:]
    tmp_beta = beta[index:]
    tmp_B = diagonal_matrix(tmp_beta)

    # calculate the sum of X and calculate R' = R - x^T \dot beta_x \dot x
    tmp_R = new_R(X, index, alpha, beta, R)

    # calculate the relaxing optimal using \sqrt(tmp_R \alpha^T dot B^-1 dot \alpha)
    ABA = np.dot(np.transpose(tmp_alpha), np.linalg.inv(tmp_B))
    ABA = np.dot(ABA, tmp_alpha)

    S = (tmp_R * ABA) ** 0.5

    return S + np.dot(np.transpose(alpha[:index]), tmp_x)


def Search(alpha, beta, R, index, X, upperbound):
    global count
    global lower_bound
    global optimal_S

    i = -1

    # recursive searching until we get to the leaf
    if index != alpha.shape[0]:
        while (i <= upperbound[index-1]):
            i += 1
            X[index-1][0] = i
            if new_R(X, index, alpha, beta, R) <= 0:
                return

            S = relaxing_optimal(X, index, alpha, beta, R)

            if S <= lower_bound:
                continue
            else:
                Search(alpha, beta, R, index+1, X, upperbound)

    # we are at the leaf node
    else:
        i = upperbound[-1] + 1
        while True:
            i += -1
            X[index - 1][0] = i

            #check whether it satisfies the constraint
            if new_R(X, index, alpha, beta, R) <= 0:
                continue

            S = np.dot(np.transpose(alpha), X)
            count += 1

            if S > lower_bound:
                lower_bound = int(S)
                optimal_S = np.copy(X)
                print("New Maximum: ", end='')
                print(lower_bound)
                print("new X: ", end='')
                print(np.transpose(X))
                print(' Counts = ' + str(count))
                print('============================================================================================')
            else:
                return

R = 68644

# the list for alpha and beta
alpha = np.array([104, 128, 135, 139, 150, 153, 162, 168, 195, 198])
beta = np.array([9, 8, 7, 7, 6, 6, 5, 2, 1, 1])

# construct the matrix B
B = diagonal_matrix(beta)

# calculating the alpha^T B alpha because it will appear a lot of times
ABA = np.dot(np.transpose(alpha), np.linalg.inv(B))
ABA = np.dot(ABA, alpha)

# calculate the optimal real u
optimal_real_u = (ABA / (4 * R)) ** 0.5

# calculate the lower bound by rounding down
x = np.dot(np.linalg.inv(B), alpha) / (2 * optimal_real_u)
x = np.array(x, dtype='int')
lower_bound = np.dot(np.transpose(alpha), x)
optimal_S = x

# calculate the upper bound for each x_i:
l = []
for i in beta:
    tmp = int((R/i) ** 0.5)
    l.append(tmp)
upperbound = np.array(l)

n = 10
X = np.reshape(np.array([0] * n), (n, 1))
Search(alpha, beta, R, 1, X, upperbound)



