import tensorflow.compat.v1 as tf
import tensorflow


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def conv_layer(input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True, bias=True):
    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[
                         1, 1, 1, 1], padding=padding)

    if bias:
        biases = create_biases(num_filters)
        layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer


def pool_layer(input, padding='SAME'):
    return tf.nn.max_pool(value=input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding=padding)


def un_conv(input, num_input_channels, conv_filter_size, num_filters, feature_map_h, feature_map_w, padding='SAME', relu=True, bias=True):

    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                   output_shape=[
                                       tf.shape(input)[0], feature_map_h, feature_map_w, num_filters],
                                   strides=[1, 2, 2, 1],
                                   padding=padding)
    if bias:
        biases = create_biases(num_filters)
        layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer


def tf_bn(name, input_layer=None, training=True):

    with tf.variable_scope(name) as scope:
        bn = tf.layers.batch_normalization(
            input_layer,
            training=training)
        return bn


def tf_conv(name, input_layer, channels_in, filters, k_height, k_width, kernel_initializer=None, d_height=1, d_width=1, mode='SAME'):

    with tf.variable_scope(name):
        if kernel_initializer is None:
            kernel_initializer = tf.variance_scaling_initializer()

        conv = tf.layers.conv2d(input_layer,
                                filters,
                                [k_height, k_width],
                                [d_height, d_width],
                                mode,
                                'channels_last',
                                kernel_initializer=kernel_initializer,
                                use_bias=True)

        x = name.split("_")[1]
        bn = tf_bn("batchnorm_"+x, conv)

        relu = tf.nn.relu(bn)
        return relu


def tf_pool(name, input_layer, k_height=2, k_width=2, d_height=2, d_width=2, mode='SAME'):
    return tf.layers.max_pooling2d(
        input_layer,
        [k_height, k_width],
        [d_height, d_width],
        padding=mode,
        data_format='channels_last',
        name=name)


def tf_upconv(name, input_layer, channels_in, filters, k_height, k_width, kernel_initializer=None, d_height=1, d_width=1, mode='SAME'):

    with tf.variable_scope(name):
        if kernel_initializer is None:
            kernel_initializer = tf.variance_scaling_initializer()

        upconv = tf.layers.conv2d_transpose(input_layer,
                                            filters,
                                            [k_height, k_width],
                                            [d_height, d_width],
                                            mode,
                                            'channels_last',
                                            kernel_initializer=kernel_initializer,
                                            use_bias=True)

        x = name.split("_")[1]
        bn = tf_bn("batchnorm_"+x, upconv)
        relu = tf.nn.relu(bn)

        return relu
