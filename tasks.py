import random
import numpy as np
import tensorflow as tf

def generate_random_vector(size):
    return [round(random.random()) for _ in range(size)]

def generate_copy_data(n, max_length=10):
    from main import INPUT_SIZE
    inps = []
    outs = []
    for i in range(n):
        current_length = 1 + int(random.random() * max_length)
        in_tensor = []
        out_tensor = []
        for j in range(current_length):
            in_tensor.append(generate_random_vector(INPUT_SIZE))
            out_tensor.append(in_tensor[j])
        for j in range(current_length):
            in_tensor.append(np.zeros([INPUT_SIZE]))
            out_tensor.append(in_tensor[j])
        inps.append(np.array(in_tensor))
        outs.append(np.array(out_tensor))
        return inps, outs

def copy_task(session, train_op, inp, given_out):
    train_inps, train_outs = generate_copy_data(5000)
    test_inps, _ = generate_copy_data(500)
    test_outs = []
    print(np.transpose(train_inps[0]))
    # Train Loop
    for i in range(len(train_inps)):
        session.run(train_op, feed_dict={
            inp: np.transpose(train_inps[i]),
            given_out: np.transpose(train_outs[i])
        })

    # Test Loop
    for i in range(len(test_inps)):
        next_out = session.run(out, feed_dict={
            inp: test_inps[i]
        })
        test_outs.append(next_out)

    print(test_outs[0:5])
