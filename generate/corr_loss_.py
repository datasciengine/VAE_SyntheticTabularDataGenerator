import tensorflow as tf


def correlation_loss(y_true, y_pred):
    """
    Calculate the correlation loss between two tensors.
    :param y_true: Tensor of shape (batch_size, n_classes)
    :param y_pred: Tensor of shape (batch_size, n_classes)
    :return: Tensor of shape (batch_size, 1)
    """
    # Calculate the mean of the true and predicted tensors
    mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
    mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)

    # Calculate the standard deviation of the true and predicted tensors
    std_true = tf.sqrt(tf.reduce_mean(tf.square(y_true - mean_true), axis=1, keepdims=True))
    std_pred = tf.sqrt(tf.reduce_mean(tf.square(y_pred - mean_pred), axis=1, keepdims=True))

    # Calculate the correlation between the true and predicted tensors
    corr = tf.reduce_mean(tf.multiply(y_true - mean_true, y_pred - mean_pred), axis=1, keepdims=True)

    # Calculate the loss
    loss = tf.divide(corr, (std_true * std_pred) + 1e-7)

    return loss
