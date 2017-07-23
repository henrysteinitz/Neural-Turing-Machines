import random
import numpy as np

def generate_random_vector(size=8):
    v = [round(random.random) for _ in range(size)]
    return np.array(v)

def generate_copy_data(n, max_length=10):
    inps = []
    outs = []
    for i in range(n):
        inps.append(generate_random_vector())
        outs.append(np.concatenate(inps[i], inps[i]))
    return inps, outs

def copy_task(session):
    test_inps, test_outs = generate_copy_data(5000)
    train_inps, train_outs = generate_copy_data(500)
