import tensorflow as tf
import numpy as np
from control import control
from focus import focus
from tasks import copy_task

INPUT_SIZE = 8
MAX_INPUT_LENGTH = 12
MEMORY_SIZE = 20
CONTROLLER_INPUT_SIZE = 2 * INPUT_SIZE
CONTROLLER_HIDDEN_SIZE = 24
OUTPUT_SIZE = INPUT_SIZE
KEY_VECTOR_SIZE = INPUT_SIZE
KEY_STRENGTH_SIZE = 1
INTERPOLATION_GATE_SIZE = 1
SHIFTER_SIZE = 3
SHARPENER_SIZE = 1

inp = tf.placeholder(tf.float32, shape=[INPUT_SIZE, None], name='inp')
given_out = tf.placeholder(tf.float32, shape=[INPUT_SIZE, None], name='given_out')
print(given_out)

# Initial values
memory_matrix = tf.zeros([INPUT_SIZE, MEMORY_SIZE])
read_out = tf.zeros([INPUT_SIZE], dtype=tf.float32)
read_attention = tf.zeros([MEMORY_SIZE])
write_attention = tf.zeros([MEMORY_SIZE])

# Parameters
hidden_weights = tf.Variable(tf.truncated_normal([
    CONTROLLER_HIDDEN_SIZE, CONTROLLER_INPUT_SIZE + 1 # add 1 for biases
]))
output_weights = tf.Variable(tf.truncated_normal([
    OUTPUT_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))

# Read weights
r_key_vector_weights = tf.Variable(tf.truncated_normal([
     KEY_VECTOR_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
r_key_strength_weights = tf.Variable(tf.truncated_normal([
    KEY_STRENGTH_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
r_interpolation_weights = tf.Variable(tf.truncated_normal([
    INTERPOLATION_GATE_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
r_shifter_weights = tf.Variable(tf.truncated_normal([
    SHIFTER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
r_sharpener_weights = tf.Variable(tf.truncated_normal([
    SHARPENER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
# Write weights
w_key_vector_weights = tf.Variable(tf.truncated_normal([
    KEY_VECTOR_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
w_key_strength_weights = tf.Variable(tf.truncated_normal([
    KEY_STRENGTH_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
w_interpolation_weights = tf.Variable(tf.truncated_normal([
    INTERPOLATION_GATE_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
w_shifter_weights = tf.Variable(tf.truncated_normal([
    SHIFTER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
w_sharpener_weights = tf.Variable(tf.truncated_normal([
    SHARPENER_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
write_vector_weights = tf.Variable(tf.truncated_normal([
    INPUT_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))
erase_vector_weights = tf.Variable(tf.truncated_normal([
    INPUT_SIZE, CONTROLLER_HIDDEN_SIZE + 1
]))

result = tf.constant(np.array([]), shape=[OUTPUT_SIZE, 0], dtype=tf.float32)

# Build recurrent graph
i = tf.constant(0)
think_vars = [
    inp,
    i,
    read_out,
    hidden_weights,
    memory_matrix,
    read_attention,
    write_attention,
    result,
    r_key_vector_weights,
    r_key_strength_weights,
    r_interpolation_weights,
    r_shifter_weights,
    r_sharpener_weights,
    w_key_vector_weights,
    w_key_strength_weights,
    w_interpolation_weights,
    w_shifter_weights,
    w_sharpener_weights,
    write_vector_weights,
    erase_vector_weights,
    output_weights
]

def think(inp, i, read_out, hidden_weights, memory_matrix, read_attention,
    write_attention, result, r_key_vector_weights, r_key_strength_weights,
    r_interpolation_weights, r_shifter_weights, r_sharpener_weights, w_key_vector_weights,
    w_key_strength_weights, w_interpolation_weights, w_shifter_weights, w_sharpener_weights,
    write_vector_weights, erase_vector_weights, output_weights):
    # Not pretty, but we can't pass dictionaries through tf.while_loop
    # Will probably refactor
    read_weights = {
        'key_vector': r_key_vector_weights,
        'key_strength': r_key_strength_weights,
        'interpolation_gate': r_interpolation_weights,
        'shifter': r_shifter_weights,
        'sharpener': r_sharpener_weights
    }
    write_weights ={
        'key_vector': w_key_vector_weights,
        'key_strength': w_key_strength_weights,
        'interpolation_gate': w_interpolation_weights,
        'shifter': w_shifter_weights,
        'sharpener': w_sharpener_weights,
        'write_vector': write_vector_weights,
        'erase_vector': erase_vector_weights
    }

    read_controls, write_controls, write_vector, erase_vector, out = \
        control(inp[i, :], read_out, hidden_weights, read_weights,
            write_weights, output_weights)
    result = tf.concat([result, tf.reshape(out, [-1, 1])], 1)

    read_attention = focus(memory_matrix, read_attention,
        read_controls['key_vector'], read_controls['key_strength'],
        read_controls['interpolation_gate'], read_controls['shifter'],
        read_controls['sharpener'])
    write_attention = focus(memory_matrix, write_attention,
        write_controls['key_vector'], write_controls['key_strength'],
        write_controls['interpolation_gate'], write_controls['shifter'],
        write_controls['sharpener'])

    read_out = tf.reshape(
        tf.matmul(memory_matrix, tf.reshape(read_attention, [-1, 1])),
        [8]
    )
    memory_matrix = memory_matrix * (1 - tf.matmul(
        tf.reshape(erase_vector, [INPUT_SIZE, 1]),
        tf.reshape(write_attention, [1, MEMORY_SIZE]),
    ))
    memory_matrix += tf.matmul(
        tf.reshape(write_vector, [INPUT_SIZE, 1]),
        tf.reshape(write_attention, [1, MEMORY_SIZE])
    )
    i += 1

    return [
        inp,
        i,
        read_out,
        hidden_weights,
        memory_matrix,
        read_attention,
        write_attention,
        result,
        r_key_vector_weights,
        r_key_strength_weights,
        r_interpolation_weights,
        r_shifter_weights,
        r_sharpener_weights,
        w_key_vector_weights,
        w_key_strength_weights,
        w_interpolation_weights,
        w_shifter_weights,
        w_sharpener_weights,
        write_vector_weights,
        erase_vector_weights,
        output_weights
    ]

def think_check(inp, i, *_):
    return i < 2 * tf.shape(inp)[1]


results = tf.while_loop(
    think_check,
    think,
    loop_vars=think_vars,
    shape_invariants=[
        inp.get_shape(),
        i.get_shape(),
        read_out.get_shape(),
        hidden_weights.get_shape(),
        memory_matrix.get_shape(),
        read_attention.get_shape(),
        write_attention.get_shape(),
        tf.TensorShape([INPUT_SIZE, None]),
        r_key_vector_weights.get_shape(),
        r_key_strength_weights.get_shape(),
        r_interpolation_weights.get_shape(),
        r_shifter_weights.get_shape(),
        r_sharpener_weights.get_shape(),
        w_key_vector_weights.get_shape(),
        w_key_strength_weights.get_shape(),
        w_interpolation_weights.get_shape(),
        w_shifter_weights.get_shape(),
        w_sharpener_weights.get_shape(),
        write_vector_weights.get_shape(),
        erase_vector_weights.get_shape(),
        output_weights.get_shape(),
    ]
)
print(given_out)
out = results[7]
# Choose optimizer and compute loss
optimizer = tf.train.GradientDescentOptimizer(.01)
loss = tf.reduce_mean(tf.square(out - given_out))
train_op = optimizer.minimize(loss) # We might have to break this up for gradient clipping

# Run tasks
session = tf.Session()
copy_task(session, train_op, inp, given_out)
session.close()
