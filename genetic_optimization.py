"""
Performs a genetic algorithm to find the optimal weights.
"""

import numpy as np
from numpy.random import randint
from numpy.random import rand
from generate_outcomes import analyze, create_outcomes
from loky import get_reusable_executor

DATA_MODEL = "fasttext"
SIZE = len(np.load(DATA_MODEL + "/data.npy", allow_pickle=True))
executor = get_reusable_executor(max_workers=4)

def objective(x):
    """
    Evaluates the outcome of a decoded bitstring.
    """
    return analyze(create_outcomes(SIZE, DATA_MODEL, (x[0], x[1])))

def decode(bounds, n_bits, bitstring):
    """
    Decodes a bitstring into real values based on provided bounds.
    """
    decoded = list()
    largest = 2**n_bits
    
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars,2)

        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
    
    return decoded

def selection(pop, scores, k=3):
    """
    Selects a candidate from the population using tournament selection.
    """
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
            
    return pop[selection_ix]

def crossover(p1, p2, r_cross):
    """
    Performs crossover between two parents to generate offspring.
    """
    c1, c2 = p1.copy(), p2.copy()
    
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        
    return [c1, c2]

def mutation(bitstring, r_mut):
    """
    Mutates a bitstring based on the mutation rate.
    """
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1-bitstring[i]
            
def genetic_algorithm(bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    """
    Runs the genetic algorithm to optimize the objective function.
    """
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    
    for gen in range(n_iter):
        
        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = list(executor.map(objective, decoded, chunksize=16))
        for i in range(n_pop):
            if scores[i] >= best_eval:
                best, best_eval = pop[i], scores[i]     
                print(">%d, new best f(%s) = %f" % (gen, decoded[i], scores[i]))
                
        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
                    
        pop = children
    return [best, best_eval]

bounds = [[0, 1], [0, 1]]
n_iter = 200
n_bits = 16
n_pop = 64
r_cross = 0.9
r_mut = 1.0 / (float(n_bits) * len(bounds))
best, score = genetic_algorithm(bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print("DONE")

decoded = decode(bounds, n_bits, best)
print((decoded, score))
