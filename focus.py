import tensorflow as tf
from helpers import cosine_sim
from functools import partial

def content_focus(memory_matrix, key_vector, key_strengh):
    sim_vector = key_strengh * tf.map_fn(partial(cosine_sim, key_vector), memory_matrix)
    return tf.nn.softmax(sim_vector)

def interpolate(read_interpolation_gate, w_c, prev_w):
    return interpolation_gate * w_c + (1 - interpolation_gate) * prev_w

def shift(w_g, shifter, shift_rule=[-1, 0, 1]):
    w_sftd = tf.zeros(w_g.shape())
    for i, n in enumerate(shift_rule):
        w_sftd += shifter[i] * roll(w_g, n)
    return w_sftd

def sharpen(w_sftd, sharpener):
    w = tf.map_fn(lambda x: tf.pow(x, sharpener), w_sftd)
    return w / tf.sum(w)

def focus(memory_matrix, prev_w, key_vector, key_strength, interpolation_gate,
    shift, sharpener):
    w_c = content_focus(memory_matrix, key_vector, key_strength)
    w_g = interpolate(interpolation_gate, w_c, prev_w)
    w_sftd = shift(w_g, shifter)
    return sharpen(w_sftd, sharpener)
