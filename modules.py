from math import cosine_sim
from functools import partial

def control(inp, read_out):
    # Emits the following:
    # read_key_vector, write_key_vector,
    # read_key_strength, write_key_strength,
    # read_interpolation_gate, write_interpolation_gate,
    # read_shift_weighting, write_shift_weighting,
    # read_sharpener, write_sharpener,
    # output
    controller_input = tf.concat(inp, read_out)
    hidden_activation = tf.nn.sigmoid(tf.matmul(controller_input, controller_weights_1))
    controller_output = tf.nn.sigmoid(tf.matmul(hidden_activation, controller_weights_2))
    return = tf.split(controller_output, [
    # read_key_vector, write_key_vector,
    # read_key_strength, write_key_strength,
    # read_interpolation_gate, write_interpolation_gate,
    # read_shift_weighting, write_shift_weighting,
    # read_sharpener, write_sharpener,
    # output
    ])

def content_focus(memory_matrix, key_vector, key_strengh):
    sim_vector = key_strengh * tf.map_fn(partial(cosine_sim, key_vector), memory_matrix)
    return tf.nn.softmax(sim_vector)

def interpolate(read_interpolation_gate, w_c_r, prev_w_r,
                write_interpolation_gate, w_c_w, prev_w_w):
    w_g_r = read_interpolation_gate * w_c_r +
            (1 - read_interpolation_gate) * prev_w_r
    w_g_w = write_interpolation_gate * w_c_w +
            (1 - write_interpolation_gate) * prev_w_w
    return w_g_r, w_g_w

def shift(w_g_r, read_shift_weighting, w_g_w, write_shift_weighting):

def sharpen(w_shift_r, read_sharpener, w_shift_w, write_sharpener):


def focus(memory_matrix, outputs, prev_w_r, prev_w_w):
    read_key_vector, write_key_vector,
    read_key_strength, write_key_strength,
    read_interpolation_gate, write_interpolation_gate,
    read_shift_weighting, write_shift_weighting,
    read_sharpener, write_sharpener, _ = outputs

    w_c_r = content_focus(memory_matrix, read_key_vector, read_key_strength)
    w_c_w = content_focus(memory_matrix, write_key_vector, write_key_strength)

    w_g_r, w_g_w = interpolate(read_interpolation_gate, w_c_r, prev_w_r,
        write_interpolation_gate, w_c_w, prev_w_w)
