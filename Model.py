import tensorflow as tf


def fc_layer(inp, in_size, out_size, activation_function=None, name="fc"):
    """
    :param inp: input tensor (rank 2, [batch_size, in_size])
    :param in_size: number of nodes in the previous layer (int)
    :param out_size: number of nodes in the current layer (int)
    :param activation_function: self explanatory (str)
    :param name: self explanatory (str)
    :return: output tensor (rank 2, [batch_size, out_size])
    """
    with tf.name_scope(name):
        with tf.variable_scope(name):
            W = tf.get_variable('W', shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', shape=[out_size], initializer=tf.contrib.layers.xavier_initializer())
        z = tf.matmul(inp, W) + b

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("z", z)

        if activation_function == 'tanh':
            activation = tf.nn.tanh(z, name='tanh_activation')
        elif activation_function == 'sigmoid':
            activation = tf.nn.sigmoid(z, name='sigmoid_activation')
        elif activation_function == 'relu':
            activation = tf.nn.relu(z, name='relu_activation')
        else:
            return z

        tf.summary.histogram("activation", activation)

        return activation


def conv_layer(inp, ch_in, ch_out, kernel_size, stride, padding="VALID", activation_function=None, name="conv"):
    """
    :param inp: the input to the layer (tensor with rank 4, 4th dimention should be ch_in)
    :param ch_in: number of input channels (int)
    :param ch_out: number of output channels (int)
    :param kernel_size: size of the the kernel (int)
    :param stride: self explanatory (int)
    :param padding: self explanatory (int)
    :param activation_function: self explanatory (int)
    :param name: tensorflow name scope and variable scope (str)
    :return: output after the convolution is applied (tensor with rank 4, 4th dimention will be ch_out, 1st one will be
    the same as the input and the 2nd and 3rd will depend on the input size, stride and padding)
    """
    with tf.name_scope(name):
        with tf.variable_scope(name):
            W = tf.get_variable('W', shape=[kernel_size, kernel_size, ch_in, ch_out],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', shape=[ch_out], initializer=tf.contrib.layers.xavier_initializer())
        z = tf.nn.conv2d(inp, W, strides=[1, stride, stride, 1], padding=padding)

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("z", z)

        if activation_function == 'tanh':
            activation = tf.nn.tanh(z, name='tanh_activation')
        elif activation_function == 'sigmoid':
            activation = tf.nn.sigmoid(z, name='sigmoid_activation')
        elif activation_function == 'relu':
            activation = tf.nn.relu(z, name='relu_activation')
        else:
            return z

        tf.summary.histogram("activation", activation)

        return activation


def model(inp):

    act = conv_layer(inp, 4, 32, 8, 4, activation_function='relu', name='conv_1')
    act = conv_layer(act, 32, 64, 4, 2, activation_function='relu', name='conv_2')
    act = conv_layer(act, 64, 64, 3, 1, activation_function='relu', name='conv_3')

    flat = tf.layers.flatten(act)

    act = fc_layer(flat, 3136, 512, activation_function='relu', name='fc_1')
    act = fc_layer(act, 512, 2, activation_function='tanh', name='fc_2')

    return act
