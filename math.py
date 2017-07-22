import numpy as np

def cosine_sim(x, y):
    return tf.tensordot(x, y) / (tf.norm(x) * tf.norm(y))
