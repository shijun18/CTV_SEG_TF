from tensorflow.python.framework import graph_util
from unet import unet
import tensorflow.compat.v1 as tf

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


with tf.Graph().as_default() as graph:

    net = unet(channels=1,n_class=2, input_shape=(256,256))
    print('stats before freezing')
    stats_graph(graph)

