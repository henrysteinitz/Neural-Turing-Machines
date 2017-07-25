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
        inps[i] = np.concatenate(inps[i], np.zeros[len(inps[i])])
    return inps, outs

def copy_task(session, train_op):
    train_inps, train_outs = generate_copy_data(5000)
    test_inps, _ = generate_copy_data(500)
    test_outs = []

    # Train Loop
    for in range(len(train_inps)):
        session.run(train_op, feed_dict={
            inp: train_inps[i],
            given_out: train_outs[i]
        })

    # Test Loop
    for in range(len(test_inps)):
        next_out = session.run(out, feed_dict={
            inp: test_inps[i]
        })
        test_outs.append(next_out)

    print(test_outs[0:5])
