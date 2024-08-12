"""
Methods used by AI to calculat similarity, and generate the following priority queues.
"""


import numpy as np
from numba import jit
import heapq
import itertools

@jit
def calcDensity(arr, adj):
    """
    Calculate the density of connections within a subset of nodes in the adjacency matrix.
    """
    a = 0

    for i in arr:
        for j in arr:
            a += adj[i][j]

    return a / (len(arr) * len(arr))


@jit
def calcConductance(arr, adj):
    """
    Calculate the conductance of a subset of nodes in the adjacency matrix.
    """
    outside = 0
    inside = 0

    for i in arr:
        for j in range(len(adj)):

            if adj[i][j] == -1:
                continue

            if j in arr:
                inside += adj[i][j] / 4
            else:
                outside += adj[i][j]

    if outside == 0 or inside == 0:
        return -1

    return 1 - outside / ((2 * inside) + outside)


def push(pq, id, num):
    """
    Push an element onto a priority queue with the specified priority.
    """
    heapq.heappush(pq, (-num, id))


def pop(pq):
    """
    Pop the highest-priority element from a priority queue.
    """
    num, id = heapq.heappop(pq)
    return id, -num


def linkPq(arr, adj, weights):
    """
    Generate a priority queue of trios linked to the given subset of nodes, based on conductance and density.
    """
    pq = []
    combinations = itertools.combinations(arr, 3)

    for combo in combinations:
        combo = list(combo)
        push(
            pq,
            combo,
            (weights[0] * calcConductance(combo, adj))
            + (weights[1] * calcDensity(combo, adj)),
        )

    return pq


def childPq(arr, adj, avail, weights):
    """
    Generate a priority queue of child nodes linked to the given subset, based on conductance and density.
    """
    pq = []

    for i in avail:
        if i not in arr:
            combo = arr + [i]
            push(
                pq,
                combo,
                (weights[0] * calcConductance(combo, adj))
                + (weights[1] * calcDensity(combo, adj)),
            )

    pop(pq)

    return pq


def genPq(adj, avail, weights):
    """
    Generate a priority queue for all combinations of four available nodes, based on conductance and density.
    """
    pq = []
    combinations = itertools.combinations(avail, 4)

    for combo in combinations:
        combo = list(combo)
        push(
            pq,
            combo,
            (weights[0] * calcConductance(combo, adj))
            + (weights[1] * calcDensity(combo, adj)),
        )

    return pq


@jit
def purge(adj, indices):
    """
    Remove connections in the adjacency matrix for a given set of indices.
    """
    for index in indices:
        for i in range(16):
            adj[index][i] = -1
            adj[i][index] = -1

    return adj


def check(lis):
    """
    Check if the input list corresponds to one of the answers.
    """
    sets = [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}]
    input = set(lis)

    for s in sets:
        if input == s:
            return 1
        elif len(input.intersection(s)) == 3:
            return 0

    return -1


def play(adj_code, word_data, weights):
    """
    Main function to simulate the AI, returning the number of turns taken.
    """
    if weights == (0, 0):
        return 99

    adj = np.load(word_data + "/" + "data.npy", allow_pickle=True)[adj_code]
    avail = range(16)
    turns = 0

    while len(avail) != 0:

        out = -1
        curr = []
        pq = genPq(adj, avail, weights)

        while out == -1:
            curr = pop(pq)[0]
            out = check(curr)
            turns += 1

        if out == 0:
            trios = linkPq(curr, adj, weights)
            out = -1

            while out != 1:
                bestTrio = pop(trios)[0]
                pq = childPq(bestTrio, adj, avail, weights)
                curr = pop(pq)[0]
                out = check(curr)
                turns += 1

                while out == 0:
                    curr = pop(pq)[0]
                    out = check(curr)
                    turns += 1

            else:
                adj = purge(adj, curr)
                avail = list(set(avail) - set(curr))

        else:
            adj = purge(adj, curr)
            avail = list(set(avail) - set(curr))

    return turns - 4

WEIGHTS = (0.70196533203125, 0.05657958984375)

#print(play(189, "fasttext", WEIGHTS))
