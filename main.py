import tensorflow as tf
import numpy as np
from math import cosine_sim
from modules import control

INPUT_WIDTH = 8
CONTROLLER_INPUT_WIDTH = 2 * INPUT_WIDTH
CONTROLLER_HIDDEN_WIDTH = 24
CONTROLLER_OUTPUT_WIDTH =
KEY_VECTOR_WIDTH = INPUT_WIDTH

MEMORY_HEIGHT = 100

# Feedforward Controller
controller_weights_1 = tf.Variable(tf.truncated_normal([
    CONTROLLER_HIDDEN_WIDTH,
    CONTROLLER_INPUT_WIDTH
]))
controller_weights_2 = tf.Variable(tf.truncated_normal([
    CONTROLLER_OUTPUT_WIDTH,
    CONTROLLER_HIDDEN_WIDTH
]))

# Memory Matrix
memory_matrix = tf.placeholder(tf.zeros([MEMORY_WIDTH, MEMORY_HEIGHT]))


tf.while_loop():


    attention = # Make this a column vector
    erase_weights = # Make this a row vector
    write_weights = # Make this a row vector

    read_out = tf.transpose(attention * tf.transpose(memory_matrix))
    memory_matrix = memory_matrix * (1 - erase_weights * attention)
    memory_matrix = write_weights * attention
