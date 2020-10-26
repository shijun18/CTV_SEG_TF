import tensorflow.compat.v1 as tf
from layers import tf_conv, tf_pool, tf_upconv

import numpy as np

class UNet(object):

    def __init__(self, channels=1, n_class=2, input_shape=(256,256)):
        self._channels = channels
        self.n_class = n_class
        self.input_shape = [None] + list(input_shape) + [self._channels]
        self.label_shape = [None] + list(input_shape) + [self.n_class]
        self.inputs = tf.placeholder(tf.float32, shape=self.input_shape, name='input')
        self.labels = tf.placeholder(tf.float32, shape=self.label_shape, name='label')
        self.build_model()

    def build_model(self):
        with tf.variable_scope("unet", reuse=tf.AUTO_REUSE):
            self.logits = self._construct_unet()

    def _construct_unet(self):
        conv1 = tf_conv("conv_1", self.inputs, self._channels, 64, 3, 3)
        conv2 = tf_conv("conv_2", conv1, 64, 64, 3, 3)
        pool2 = tf_pool("pool_1", conv2)

        conv3 = tf_conv("conv_3", pool2, 64, 128, 3, 3)
        conv4 = tf_conv("conv_4", conv3, 128, 128, 3, 3)
        pool4 = tf_pool("pool_2", conv4)

        conv5 = tf_conv("conv_5", pool4, 128, 256, 3, 3)
        conv6 = tf_conv("conv_6", conv5, 256, 256, 3, 3)
        pool6 = tf_pool("pool_3", conv6)

        conv7 = tf_conv("conv_7", pool6, 256, 512, 3, 3)
        conv8 = tf_conv("conv_8", conv7, 512, 512, 3, 3)
        pool8 = tf_pool("pool_3", conv8)

        conv9 = tf_conv("conv_9", pool8, 512, 1024, 3, 3)
        conv10 = tf_conv("conv_10", conv9, 1024, 1024, 3, 3)

        conv11 = tf_upconv("upconv_1", conv10, 1024, 512,
                           2, 2, d_height=2, d_width=2)
        merge11 = tf.concat(values=[conv11, conv8], axis=-1, name="concat_1")

        conv12 = tf_conv("conv_11", merge11, 1024, 512, 3, 3)
        conv13 = tf_conv("conv_12", conv12, 512, 512, 3, 3)

        conv14 = tf_upconv("upconv_2", conv13, 512, 256,
                           2, 2, d_height=2, d_width=2)
        merge14 = tf.concat([conv14, conv6], axis=-1, name="concat_2")

        conv15 = tf_conv("conv_13", merge14, 512, 256, 3, 3)
        conv16 = tf_conv("conv_14", conv15, 256, 256, 3, 3)

        conv17 = tf_upconv("upconv_3", conv16, 256, 128,
                           2, 2, d_height=2, d_width=2)
        merge17 = tf.concat([conv17, conv4], axis=-1, name="concat_3")

        conv18 = tf_conv("conv_15", merge17, 256, 128, 3, 3)
        conv19 = tf_conv("conv_16", conv18, 128, 128, 3, 3)

        conv20 = tf_upconv("upconv_4", conv19, 128, 64,
                           2, 2, d_height=2, d_width=2)
        merge20 = tf.concat([conv20, conv2], axis=-1, name="concat_4")

        conv21 = tf_conv("conv_17", merge20, 128, 64, 3, 3)
        conv22 = tf_conv("conv_18", conv21, 64, 64, 3, 3)
        conv23 = tf_conv("conv_19", conv22, 64, self.n_class, 1, 1)

        return conv23


def unet(name='unet',**kwargs):
    net = UNet(**kwargs)
    return net


if __name__ == "__main__":
    import os
    tf.disable_eager_execution()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    net = unet(channels=1,n_class=2, input_shape=(512,512))
    tf.logging.info(str(net.logits.shape))

    images = np.random.random((1,512,512,1)).astype(np.float32)
    # labels = np.ones((1,3),dtype=np.uint8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits = sess.run([net.logits],feed_dict={net.inputs:images})
        # print(logits.shape)
    total_parameters = 0
    for variable in tf.trainable_variables():
        print(variable.name)
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= int(dim)
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
