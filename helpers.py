import tensorflow as tf

def cosine_sim(x, y):
    return tf.tensordot(x, y, axes=1) / (tf.norm(x) * tf.norm(y))

def roll(x, n):
    from main import MEMORY_SIZE
    k = 20 % (-1 * n)
    return tf.concat([tf.slice(x, [k], [-1]), tf.slice(x, [0], [k])], 0)
