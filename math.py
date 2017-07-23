import tensorflow as tf

def cosine_sim(x, y):
    return tf.tensordot(x, y) / (tf.norm(x) * tf.norm(y))

def roll(x, n):
    k = x.shape()[0] % (-1 * k)
    return tf.concat(tf.slice(x, [k], [-1]), tf.slice(x, [0], k))
