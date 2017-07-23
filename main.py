import tensorflow as tf
import numpy as np
from control import control
from focus import foucus
from tasks import copy_task

INPUT_SIZE = 8
MEMORY_SIZE = 20
CONTROLLER_INPUT_SIZE = 2 * INPUT_WIDTH
CONTROLLER_HIDDEN_SIZE = 24
KEY_VECTOR_SIZE = INPUT_SIZE
KEY_STRENGTH_SIZE = 1
SHIFTER_SIZE = 3
SHARPENER_SIZE = 1

session = tf.Session()
inp = tf.placeholder([INPUT_SIZE, None])

# Initial values
memory_matrix = tf.zeros([INPUT_SIZE, MEMORY_SIZE]))
read_out = tf.zeros([INPUT_SIZE])
read_attention = tf.zeros([MEMORY_SIZE])
write_attention = tf.zeros([MEMORY_SIZE])

# Parameters
hidden_weights = tf.Variable(tf.truncated_normal([
    CONTROLLER_HIDDEN_SIZE, CONTROLLER_INPUT_SIZE + 1 # add 1 for biases
]))
read_weights = {
    'key_vector': tf.Variable(tf.truncated_normal([
         KEY_VECTOR_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'key_strength': tf.Variable(tf.truncated_normal([
        KEY_STRENGTH_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'shifter': tf.Variable(tf.truncated_normal([
        SHIFTER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'sharpener': tf.Variable(tf.truncated_normal([
        SHARPENER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
}
write_weights = {
    'key_vector': tf.Variable(tf.truncated_normal([
        KEY_VECTOR_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'key_strength': tf.Variable(tf.truncated_normal([
        KEY_STRENGTH_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'shifter': tf.Variable(tf.truncated_normal([
        SHIFTER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'sharpener': tf.Variable(tf.truncated_normal([
        CONTROLLER_HIDDEN_SIZE, SHARPENER_SIZE + 1
    ])),
    'write_vector': tf.Variable(tf.truncated_normal([
        INPUT_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
    'erase_vector': tf.Variable(tf.truncated_normal([
        INPUT_SIZE, CONTROLLER_HIDDEN_SIZE + 1
    ])),
}

# Build recurrent graph
tf.while_loop():
    read_controls, write_controls, write_vector, erase_vector, out = \
        control(inp, read_out, hidden_weights, read_weights, write_weights)
    read_attention = focus(memory_matrix, read_attention, \
        read_controls['key_vector'], read_controls['key_strength'], \
        read_controls['shifter'], read_controls['sharpener'])
    write_attention = focus(memory_matrix, write_attention, \
        read_controls['key_vector'], write_controls['key_strength'], \
        read_controls['shifter'], read_controls['sharpener'])

    read_out = tf.transpose(read_attention * tf.transpose(memory_matrix))
    memory_matrix = memory_matrix * (1 - erase_vector * write_attention)
    memory_matrix += write_vector * attention

# Choose Optimizer
optimizer = tf.train.GradientDescentOptimizer()

# Run tasks
copy_task(session)
session.close()
