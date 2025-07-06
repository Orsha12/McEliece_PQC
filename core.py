import random

import numpy as np
import galois
from sklearn.metrics import pairwise_distances
from itertools import combinations

GF2 = galois.GF(2)

# TESTS:
############################################## EXAMPLE 0 #######################################################
# based on example 1 Lect.13 p.4
A_0 = GF2([[1, 1, 1], [1, 1, 0]])
S_0 = GF2([[0, 1], [1, 0]])
P_0 = GF2([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])
EXAMPLE_0 = [A_0, S_0, P_0]
############################################## EXAMPLE 1 #######################################################
# based on example A matrix from lecture 3. The dimensions of the matrices match what Meir requested.
A_1 = GF2([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

S_1 = GF2([
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
])  # Left transformation nonsingular matrix (k x k)
# Check if the rank is full (i.e., equal to the number of rows)
# is_invertible = np.linalg.det(S_4) != 0
# print("Is the matrix invertible over GF(2)?", is_invertible)
P_1 = GF2([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])  # Permutation matrix (n x n)
EXAMPLE_1 = [A_1, S_1, P_1]

# 111111111111
# 11100000000000000000000
############################################## EXAMPLE 2 #######################################################
# An example designed using Chat GPT
A_2 = GF2([
    [1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0, 1]
])

S_2 = GF2([
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 0, 0]
])  # Left transformation nonsingular matrix (k x k)

P_2 = GF2([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])  # Permutation matrix (n x n)
EXAMPLE_2 = [A_2, S_2, P_2]

############################################## EXAMPLE 3 #######################################################

A_3 = GF2([
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
])

S_3 = GF2([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
])

P_3 = GF2([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

EXAMPLE_3 = [A_3, S_3, P_3]

def find_minimum_hamming_distance(B):
    """
    Computes the minimum Hamming distance between all distinct pairs of rows in a binary matrix B.

    Parameters:
        B (ndarray): A binary matrix (2D NumPy array) where each row represents a codeword.

    Returns:
        int: The minimal Hamming distance between any two different rows in B.
    """
    # Compute pairwise Hamming distances (normalized between 0 and 1), then scale by number of bits
    D = (pairwise_distances(B, metric='hamming') * B.shape[1]).astype(int)

    # Set diagonal to maximum distance to ignore self-comparisons
    np.fill_diagonal(D, B.shape[1])

    # Return the minimal distance among all pairs
    return int(np.min(D))


def get_binary_vector_from_user(length, num_1s=-1):
    """
    Prompts the user to input a binary vector of a given length, optionally enforcing an exact number of ones.

    Parameters:
        length (int): Required length of the vector.
        num_1s (int): Exact number of ones required. If -1, any number of ones is allowed.

    Returns:
        ndarray: Binary vector as numpy array of 0s and 1s.
    """
    while True:
        prompt = f"Enter a binary vector of length {length}, using only 0 and 1"
        if num_1s != -1:
            prompt += f" with exactly {num_1s} ones"
        prompt += " (e.g. 101001...):\n"

        user_input = input(prompt)
        user_input = user_input.strip().replace(" ", "")

        # Validate length
        if len(user_input) != length:
            print(f"Invalid input. Please enter exactly {length} bits.")
            continue

        # Validate binary characters
        if any(c not in "01" for c in user_input):
            print("Invalid input. The input vector must consist of 0s and 1s only.")
            continue

        # Convert to list of ints
        vec = GF2([np.uint8(bit) for bit in user_input])

        # Validate number of ones if specified
        if num_1s != -1 and sum(vec.tolist()) != num_1s:
            print(f"Invalid input. The vector must contain exactly {num_1s} ones.")
            continue

        return vec


def generate_error_dict(vector_length, error_num, B):
    """
    Returns a dict mapping syndrome to error vector for all errors of weight 'error_num'.
    """
    dic = {}
    for i in range(0,error_num+1,1):
        for ones_positions in combinations(range(int(vector_length)),i):
            e = GF2.Zeros(vector_length)
            e[list(ones_positions)] = 1
            s = e @ B.T
            key = tuple(s.tolist())
            if key not in dic:
                dic[key] = e
    return dic


def create_error_vector(length, errors_num):
    """
    Generates a random binary error vector of specified length with a given number of 1s (errors).

    Parameters:
        length (int): The length of the error vector.
        errors_num (int): Number of bits to flip (i.e., number of 1s in the vector).

    Returns:
        ndarray: A binary NumPy array with 'errors_num' 1s at random positions.
    """
    # Initialize a zero vector
    err = GF2.Zeros(length)

    # Randomly choose positions to flip (insert 1s)
    error_random= random.randint(0,errors_num)
    ones_positions = np.random.choice(length, size=error_random, replace=False)
    err[ones_positions] = 1

    # Return the generated error vector
    return err


class LinearCode:
    def __init__(self, A, S, P):
        self.G = GF2(np.hstack((GF2(np.eye(A.shape[0], dtype=np.uint8)), A)))
        self.H = GF2(np.hstack((A.T, GF2(np.eye(A.shape[1], dtype=np.uint8)))))
        self.S = S
        self.P = P
        self.G_tag = S @ self.G @ P
        self.k, self.n = self.G_tag.shape
        self.d_min = find_minimum_hamming_distance(self.G_tag)
        self.max_errors_num = (self.d_min - 1) // 2
        print(f"Code initialized: [n={self.n}, k={self.k}, d_min={self.d_min}]")

    def encode(self, message, error):
        plain = (message @ self.G_tag) + error
        return plain

    def decode(self, plaintext, expected_error=None, expected_syndrome=None):
        coset_leader_dict = generate_error_dict(self.n, self.max_errors_num, self.H)
        plaintext_tag = plaintext @ self.P.T
        syndrome = plaintext_tag @ self.H.T
        # Testing - Does the syndrome matches the expected one?
        if expected_syndrome is not None:
            if np.array_equal(syndrome, expected_syndrome):
                print("Syndrome matches expected one")
            else:
                print(f"Syndrome Mismatch! Expected: {expected_syndrome}, but got: {syndrome}")

        estimated_error = coset_leader_dict[tuple(syndrome.tolist())]
        # Testing - Does the error matches the expected one?
        if expected_error is not None:
            if np.array_equal(estimated_error, expected_error):
                print("Estimated error matches expected one ")
            else:
                print(f"Error mismatch. Expected: {expected_error}, but got: {estimated_error}")

        estimated_cypher_text = plaintext_tag + estimated_error
        estimated_message = (estimated_cypher_text[:self.k]) @ np.linalg.inv(self.S)
        return estimated_message

    def simulate_manual(self):
        print("Choose your message vector:")
        message = get_binary_vector_from_user(self.k)
        print("Choose your error vector:")
        error = get_binary_vector_from_user(self.n, self.max_errors_num)

        expected_estimated_error = error @ self.P.T
        expected_syndrome = expected_estimated_error @ self.H.T
        paint_text = self.encode(message, error)
        estimated_message = self.decode(paint_text, expected_estimated_error, expected_syndrome)

        # Testing - Does the estimated message matches the expected one?
        if np.array_equal(estimated_message, message):
            print(f"Estimated message matches to the original one :). the message is:{estimated_message} ")
        else:
            print(f"Message mismatch :(. Expected: {message}, but got: {estimated_message}")

    def simulate_random(self):
        message = GF2.Random(self.k)
        error = GF2(create_error_vector(self.n, self.max_errors_num))
        print(f"Random message: {message}")
        print(f"Random error: {error}")
        expected_estimated_error = error @ self.P.T
        expected_syndrome = expected_estimated_error @ self.H.T

        plaintext = self.encode(message, error)
        estimated_message = self.decode(plaintext)

        # Testing - Does the estimated message matches the expected one?
        if np.array_equal(estimated_message, message):
            print(f"Estimated message matches to the original one :). the message is:{estimated_message} ")
        else:
            print(f"Message mismatch :(. Expected: {message}, but got: {estimated_message}")
        return int(np.array_equal(message, estimated_message))


if __name__ == "__main__":

    EXAMPLES = [ EXAMPLE_1]
    examples_num = 10
    for example in range(len(EXAMPLES)):
        print(
            f"############################################## EXAMPLE {example} ##############################################################")
        A = EXAMPLES[example][0]
        S = EXAMPLES[example][1]
        P = EXAMPLES[example][2]
        code = LinearCode(A, S, P)
        success_count = 0
        for i in range(examples_num):
            print(f"################# EXAMPLE {example}.{i} ################")
            success_count += code.simulate_random()
        success_rate = (success_count / examples_num) * 100
        print(f"\nFinal success rate of test {example}: {success_count}/{examples_num} = {success_rate:.2f}%")
