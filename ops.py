import tensorflow as tf

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def conv_with_lrelu(ip, ip_channel, out_channel, scope):

    with tf.variable_scope(scope):
        filter_shape = [4,4,ip_channel, out_channel]
        W = tf.get_variable('W', initializer= tf.truncated_normal(shape= filter_shape, stddev= 0.01))
        B = tf.get_variable('B', shape=[out_channel], initializer= tf.constant_initializer(value=0.))

        conv = tf.nn.conv2d(input = ip,
                            filter = W,
                            strides = [1,2,2,1],
                            padding = 'SAME') + B

        conv_lrelu = lrelu(conv)
    return conv_lrelu

def deconv(ip, target_shape, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable('w', [5, 5, target_shape[-1], ip.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.01))
        conv_transpose = tf.nn.conv2d_transpose(value=ip,
                                                filter=W,
                                                output_shape=target_shape,
                                                strides=[1,2,2,1])
        biases = tf.get_variable('biases', [target_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(conv_transpose, biases), conv_transpose.get_shape())

        return deconv

def linear(ip, ip_size, out_size, scope):

    W = tf.get_variable('W', initializer= tf.truncated_normal(shape=[ip_size, out_size]))
    B = tf.get_variable('B', shape=[out_size], initializer=tf.constant_initializer(value=0.1))

    z = tf.matmul(ip,W)+B

    return z
