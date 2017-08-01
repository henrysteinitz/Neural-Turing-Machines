import random
import numpy as np
import tensorflow as tf

def generate_random_vector(size):
    return [round(random.random()) for _ in range(size)]

def generate_copy_data(n):
    from main import INPUT_SIZE, MAX_INPUT_LENGTH
    inps = []
    outs = []
    for i in range(n):
        current_length = 1 + int(random.random() * MAX_INPUT_LENGTH)
        in_tensor = []
        out_tensor = []
        for j in range(current_length):
            in_tensor.append(generate_random_vector(INPUT_SIZE))
            out_tensor.append(in_tensor[j])
        for j in range(current_length):
            in_tensor.append(np.zeros([INPUT_SIZE]))
            out_tensor.append(in_tensor[j])
        for j in range(2*MAX_INPUT_LENGTH - 2*current_length):
            in_tensor.append(np.zeros([INPUT_SIZE]))
            out_tensor.append(np.zeros([INPUT_SIZE]))
        inps.append(np.array(in_tensor))
        outs.append(np.array(out_tensor))
    return inps, outs

def copy_task(session, train_op, inp, given_out, out):
    train_inps, train_outs = generate_copy_data(100000)
    test_inps, true_test_outs = generate_copy_data(50)
    print(np.transpose(train_outs[0]))

    # Train Loop
    for i in range(len(train_inps)):
        session.run(train_op, feed_dict={
            inp: np.transpose(train_inps[i]),
            given_out: np.transpose(train_outs[i])
        })
        # Test Loop
        np.set_printoptions(precision=0)
        if i % 20 == 0:
            test_outs = []
            for j in range(len(test_inps)):
                next_out = session.run(out, feed_dict={
                    inp: np.transpose(test_inps[j])
                })
                test_outs.append(np.transpose(next_out))
            errors = [
                np.square(test_outs[k] - true_test_outs[k]).sum()
                for k in range(len(test_outs))
            ]
            print(sum(errors))
        if i % 100 == 0:
            print(test_outs[0])
            print(true_test_outs[0])
            print(test_outs[1])
            print(true_test_outs[1])
