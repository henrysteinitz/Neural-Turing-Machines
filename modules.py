from math import cosine_sim
from functools import partial

# ---Control Modules---

def lstm_layer(inp, weights):
    return

def feedforward_layer(inp, weights):
    return tf.nn.sigmoid(tf.matmul(inp, weights))

def key_vector_out(hidden_state, weights):
    return feedforward_layer(hidden_state, weights)

def key_strengh_out(hidden_state, weights):
    return feedforward_layer(hidden_state, weights)

def shift_weighting_out(hidden_state, weights):
    return tf.nn.softmax(tf.matmul(hidden_state, weights))

def sharpener_out(hidden_state, weights):
    return 1 / feedforward_layer(hidden_state, weights)

def control(inp, read_out, hidden_weights, kind='feedforward', io_weights):
    controller_input = tf.concat(inp, read_out)
    hidden_state = feedforward_layer(controller_input, hidden_weights)

    key_vector = key_vector_out(hidden_state, io_weights['key_vector'])
    key_strength = key_strengh_out(hidden_state, io_weights['key_strength'])

    re
    return = tf.split(controller_output, [
    # read_key_vector, write_key_vector,
    # read_key_strength, write_key_strength,
    # read_interpolation_gate, write_interpolation_gate,
    # read_shift_weighting, write_shift_weighting,
    # read_sharpener, write_sharpener,
    # output
    ])


# ---Focus Modules---

def content_focus(memory_matrix, key_vector, key_strengh):
    sim_vector = key_strengh * tf.map_fn(partial(cosine_sim, key_vector), memory_matrix)
    return tf.nn.softmax(sim_vector)

def interpolate(read_interpolation_gate, w_c, prev_w):
    return interpolation_gate * w_c + (1 - interpolation_gate) * prev_w

def shift(w_g, shift_weighting):
    # We'll use shift weightings of length 3 corresponding to [-1, 0 , 1]
    # Shift weightings are obtained by a softmax attached to the controler


def sharpen(w_sft, sharpener):


def focus(memory_matrix, prev_w, key_vector, key_strength, interpolation_gate,
    shift_weighting, sharpener):
    w_c = content_focus(memory_matrix, key_vector, key_strength)
    w_g = interpolate(interpolation_gate, w_c, prev_w)
    w_sft = shift(w_g, shift_weighting)
    return sharpen(w_sft, sharpener)
