import cmath
import numpy as np
import random

# RSK algorithm for Grothendieck polynomials

# parameters

x = 1./3
y = 1./5
beta = -6
n = 50


# interlacing array lambda of size n with indexing lambda[k][j], 1\le j\le k \le n

my_lambda = np.zeros((n+1,n+1))
for k in range(1,n+1):
    for j in range(1,k+1):
        my_lambda[k][j] = 0

# print without the first component; this is a signature of length n:
# print(my_lambda[n][1:])

def generate_geom_random_variable(alpha):
    if not 0 <= alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    # Generate a random number from U(0, 1)
    u = random.random()

    # Initialize sum of probabilities and k
    sum_prob = (1 - alpha)
    k = 0

    # Use the inverse transform method
    while u > sum_prob:
        # Move to the next value k
        k += 1
        # Calculate the probability P(ξ = k) and add it to the sum
        sum_prob += (1 - alpha) * (alpha ** k)

    return k

def generate_bern_random_variable(alpha):
    if not 0 <= alpha:
        raise ValueError("Beta must be nonnegative.")

    # Generate a random number from U(0, 1)
    u = random.random()

    if u < alpha / (1. + alpha):
        return 1
    else:
        return 0

# print first column of the array:
# print(my_bern_array[:,0])

# First, we define the operation pull as follows [this is from MP-2015]:
# Definition 4.2. (Deterministic long-range pulling, Fig. 11) Let $j=2, \ldots, N$, and signatures $\bar{\lambda}, \bar{\nu} \in \mathbb{G T}_{j-1}^{+}, \lambda \in \mathbb{G T}_j^{+}$satisfy $\bar{\lambda} \prec_{\mathrm{h}} \lambda, \bar{\nu}=\bar{\lambda}+\overline{\mathrm{e}}_i$, where $\overline{\mathrm{e}}_i=(0,0, \ldots, 0,1,0, \ldots, 0)$ (for some $i=1, \ldots, j-1)$ is the $i$ th basis vector of length $j-1$. Define $\nu \in \mathbb{G T}_j^{+}$to be
# $$
# \nu=\operatorname{pull}(\lambda \mid \bar{\lambda} \rightarrow \bar{\nu}):= \begin{cases}\lambda+\mathrm{e}_i, & \text { if } \bar{\lambda}_i=\lambda_i ; \\ \lambda+\mathrm{e}_{i+1}, & \text { otherwise }\end{cases}
# $$
# Here $\mathrm{e}_i$ and $\mathrm{e}_{i+1}$ are basis vectors of length $j$.

def pull(i, bar_lambda_sig, lambda_sig):
    #find the length of the signature lambda_sig
    j = len(lambda_sig)
    # print(j)
    nu = np.zeros(j)
    for k in range(0,j):
        nu[k] = lambda_sig[k]
    # print(nu)

    if bar_lambda_sig[i] == lambda_sig[i]:
        nu[i] = nu[i] + 1
    else:
        nu[i+1] = nu[i+1] + 1

    return nu

# Q_row(x*y), n times

for t in range(1,n+1):
    # if t % 20 == 0:
        # print(t)

    my_nu = np.zeros((n+1,n+1))

    for k in range(1,n+1):
        for j in range(1,k+1):
            my_nu[k][j] = my_lambda[k][j]
    my_nu[1][1] += generate_geom_random_variable( x*y )

    for j in range(2,n+1):
        # define a tmp array, it first copies the j-th row of my_lambda
        tmp_row = np.zeros(j+1)
        for l in range(0,j+1):
            tmp_row[l] = my_lambda[j][l]
        
        for i in range(j-1,0,-1):
            # print(" i = ", i, " j = ", j)
            c = my_nu[j-1][i] - my_lambda[j-1][i]
            # print("c = ", int(c))
            # now we do c operations pull to tmp_row, using each time bar_lambda = my_lambda[j-1] + (p-1)*e_i
            for p in range(1,int(c+1)):
                # print("p = ", p)
                # first, we define bar_lambda
                bar_lambda = np.zeros(j)
                for k in range(0,j):
                    bar_lambda[k] = my_lambda[j-1][k]
                bar_lambda[i] = bar_lambda[i] + (p-1)
                # print("tmp_row = ", tmp_row)
                tmp_row = pull(i, bar_lambda, tmp_row)
                # print("tmp_row new = ", tmp_row)
        
        # now we copy tmp_row to my_nu
        for l in range(0,j+1):
            my_nu[j][l] = tmp_row[l]
        
        my_nu[j][1] += generate_geom_random_variable( x*y )

    # now copy the whole my_nu to my_lambda
    for k in range(1,n+1):
        for j in range(1,k+1):
            my_lambda[k][j] = my_nu[k][j]

# print(my_nu)

groth_result = np.zeros(n+1)

# dual Q_row(-beta*x), n-1 times

for t in range(1,n):

    groth_result[n-t] = my_lambda[n][n-t+1]

    # if t % 20 == 0:
        # print(t)

    my_nu = np.zeros((n+1,n+1))
    
    for k in range(1,n+1):
        for j in range(1,k+1):
            my_nu[k][j] = my_lambda[k][j]

    my_nu[1][1] += generate_bern_random_variable( - beta*x )
    
    for j in range(1,n+1):
        # define a tmp array, it first copies the j-th row of my_lambda
        tmp_row = np.zeros(j+1)
        for l in range(0,j+1):
            tmp_row[l] = my_lambda[j][l]

        tmp_row[1] += generate_bern_random_variable( - beta*x )
        
        for i in range(1,j):
            # print(" i = ", i, " j = ", j)
            c = my_nu[j-1][i] - my_lambda[j-1][i]
            # print("c = ", int(c))
            # now we do c operations pull to tmp_row, using each time bar_lambda = my_lambda[j-1] + (p-1)*e_i
            if c == 1:
                tmp_row = pull(i, my_lambda[j-1], tmp_row)
        
        # now we copy tmp_row to my_nu
        for l in range(0,j+1):
            my_nu[j][l] = tmp_row[l]
        
    # now copy the whole my_nu to my_lambda
    for k in range(1,n+1):
        for j in range(1,k+1):
            my_lambda[k][j] = my_nu[k][j]

# print(my_nu)


print("{", end = "")
for i in range(1,n):
    print(int(groth_result[i]), end = ", ")
print(int(groth_result[n]), end = "")
print("}")

