import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import dispatch
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops

@dispatch.add_dispatch_support
def stateless_random_rotate(images, max_degree, seed, **kwargs):
    if max_degree < 0:
        raise ValueError('max_degree must be non-negative.')

    images = ops.convert_to_tensor(images, name='image')
    shape = images.get_shape()

    if shape.ndims is None:
        rank = array_ops.rank(images)
    else:
        rank = shape.ndims

    if rank == 2 or rank == 3:
        degrees = stateless_random_ops.stateless_random_uniform(
        shape=[], minval = -max_degree, maxval = max_degree, seed = seed)

    elif rank == 4:
        batch_size = array_ops.shape(images)[0]
        degrees = stateless_random_ops.stateless_random_uniform(
            shape=[batch_size], minval = -max_degree, maxval = max_degree, seed = seed)
    else:
        raise ValueError(
            '\'image\' (shape %s) must have either 2 (HW), 3 (HWC) or 4 (NHWC) dimensions.' % shape)

    images = tfa.image.rotate(images, degrees, **kwargs)
    images.set_shape(shape)

    return images


@dispatch.add_dispatch_support
def stateless_random_invert(images, seed):
    with ops.name_scope(None, "random_invert", [images]) as scope:
        images = ops.convert_to_tensor(images, name='image')
        shape = images.get_shape()

        if shape.ndims is None:
            rank = array_ops.rank(images)
        else:
            rank = shape.ndims

        if rank == 2 or rank == 3:
            prob = stateless_random_ops.stateless_random_uniform(
            shape=[], minval = 0., maxval = 1., seed = seed)

        elif rank == 4:
            batch_size = array_ops.shape(images)[0]
            prob = stateless_random_ops.stateless_random_uniform(
                shape=[batch_size], minval = 0., maxval = 1., seed = seed)
        else:
            raise ValueError(
                '\'image\' (shape %s) must have either 2 (HW), 3 (HWC) or 4 (NHWC) dimensions.' % shape)

        prob = tf.cast(tf.math.greater(prob, 0.5), tf.float32)
        prob = tf.reshape(prob, (batch_size, 1, 1, 1))

        images = images * tf.math.pow(-1., prob) + prob
        images.set_shape(shape)

    return images

