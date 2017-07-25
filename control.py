import tensorflow as tf

def attach_1(inp): # for biases
    return tf.concat(tf.constant([1]), inp)

def lstm_layer(inp, weights):
    return

def feedforward_layer(inp, weights):
    return tf.nn.sigmoid(tf.matmul(attach_1(inp), weights))

def key_vector_out(hidden_state, weights):
    return feedforward_layer(hidden_state, weights)

def key_strengh_out(hidden_state, weights):
    return feedforward_layer(hidden_state, weights)

def shifter_out(hidden_state, weights):
    return tf.nn.softmax(tf.matmul(attach_1(hidden_state), weights))

def sharpener_out(hidden_state, weights):
    return 1 / feedforward_layer(hidden_state, weights)

def hidden_control(inp, read_out, hidden_weights, kind):
    controller_input = tf.concat(inp, read_out)
    if kind == 'feedforward':
        hidden_state = feedforward_layer(controller_input, hidden_weights)
    elif kind == '':
        hidden_state = lstm_layer(controller_input, hidden_weights)
    else:
        raise 'Invalid controller kind: {}'.format(kind)
    return hidden_state

def io_control(hidden_state, io_weights):
    key_vector = key_vector_out(hidden_state, io_weights['key_vector'])
    key_strength = key_strengh_out(hidden_state, io_weights['key_strength'])
    shifter = shifter_out(hidden_state, io_weights['shifter'])
    sharpener = sharpener_out(hidden_state, io_weights['sharpener'])
    return {
        'key_vector': key_vector,
        'key_strength': key_strength,
        'shifter': shifter,
        'sharpener': sharpener
    }

def control(inp, read_out, hidden_weights, read_weights, write_weights, kind='feedforward'):
    hidden_state = hidden_control(inp, read_out, hidden_weights, kind)
    read_controls = io_control(hidden_state, read_weights)
    write_controls = io_control(hidden_state, write_weights)
    write_vector = feedforward_layer(hidden_state, write_weights['write_vector'])
    erase_vector = feedforward_layer(hidden_state, write_weights['erase_vector'])
    out = feedforward_layer(hidden_state, output_weights)
    return read_controls, write_controls, write_vector, erase_vector, out
