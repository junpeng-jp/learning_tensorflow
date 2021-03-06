import tensorflow as tf
import tensorflow_addons as tfa


def _random_batch_uniform(shape_tensor, seed, minval = 0, maxval = None):
    rank = shape_tensor.get_shape()

    if rank == 2 or rank == 3:
        rand_uniform = tf.random.stateless_uniform(
        shape=[], minval = minval, maxval = maxval, seed = seed)

    elif rank == 4:
        batch_size = shape_tensor[0]
        rand_uniform = tf.random.stateless_uniform(
            shape=[batch_size], minval = minval, maxval = maxval, seed = seed)
    else:
        raise ValueError(
            '\'image\' (shape %s) must have either 2 (HW), 3 (HWC) or 4 (NHWC) dimensions.' % shape)

    return rand_uniform


def stateless_random_rotate(images, max_degree, seed, **kwargs):
    if max_degree < 0:
        raise ValueError('max_degree must be non-negative.')

    images = tf.convert_to_tensor(images, name='image')
    static_shape = images.get_shape()
    shape = tf.shape(images)

    degrees = _random_batch_uniform(shape, seed = seed, minval = -max_degree, maxval = max_degree)

    images = tfa.image.rotate(images, degrees, **kwargs)
    tf.ensure_shape(images, static_shape)

    return images


def stateless_random_invert(images, seed, prob_threshold = 0.5):
    
    images = tf.convert_to_tensor(images, name='image')
    static_shape = images.get_shape()
    shape = tf.shape(images)

    prob = _random_batch_uniform(shape, seed = seed, minval = 0., maxval = 1.)

    prob = tf.where(prob > prob_threshold, 1., 0.)
    prob = tf.reshape(prob, (shape[0], 1, 1, 1))

    with tf.name_scope("random_invert") as scope:
        images = images * tf.math.pow(-1., prob) + prob
        
    tf.ensure_shape(images, static_shape)

    return images

