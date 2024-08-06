import tensorflow as tf


def tf_categorical_accuracy(pred, truth):
    """Accuracy metric"""
    return tf.reduce_mean(tf.cast(tf.equal(pred, truth), dtype=tf.float32))


def tf_categorical_dice(pred, truth, k):
    """Dice overlap metric for label k"""
    A = tf.cast(tf.equal(pred, k), dtype=tf.float32)
    B = tf.cast(tf.equal(truth, k), dtype=tf.float32)
    return 2 * tf.reduce_sum(tf.multiply(A, B)) / (tf.reduce_sum(A) + tf.reduce_sum(B))
