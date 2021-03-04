import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import dispatch
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops

@dispatch.add_dispatch_support
def stateless_random_rotate(image, max_degree, seed, **kwargs):
    if max_degree < 0:
        raise ValueError('max_degree must be non-negative.')

    image = ops.convert_to_tensor(image, name='image')
    shape = image.get_shape()

    if shape.ndims is None:
        rank = array_ops.rank(image)
    else:
        rank = shape.ndims

    if rank == 2 or rank == 3:
        degrees = stateless_random_ops.stateless_random_uniform(
        shape=[], minval = -max_degree, maxval = max_degree, seed = seed)

    elif rank == 4:
        batch_size = array_ops.shape(image)[0]
        degrees = stateless_random_ops.stateless_random_uniform(
            shape=[batch_size], minval = -max_degree, maxval = max_degree, seed = seed)
    else:
      raise ValueError(
          '\'image\' (shape %s) must have either 2 (HW), 3 (HWC) or 4 (NHWC) dimensions.' % shape)

    image = tfa.image.rotate(image, degrees, **kwargs)
    image.set_shape(shape)

    return image