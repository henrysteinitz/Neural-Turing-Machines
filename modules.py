def control(inp, read_out):
    controller_input = tf.concat(inp, read_out)
    hidden_activation = tf.nn.sigmoid(tf.matmul(controller_input, controller_weights_1))
    controller_output = tf.nn.sigmoid(tf.matmul(hidden_activation, controller_weights_2))
    key_vector,
    key_strengh,
    interpolation_gate,
     = tf.split(controller_output, [])
    return []
