import tensorflow as tf
import numpy as np


class TextMLP(object):
    """
    A MLP for text classification.
    Uses linear layers, RelU layers and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      filter_sizes, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        # Create a linear + RelU layer for each filter size
        h_size = sequence_length
        h = self.input_x
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("linear-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [h_size, filter_size]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name="b")
                lin = tf.matmul(h, W) + b
                # Apply nonlinearity
                h = tf.nn.relu(lin, name="relu")
                with tf.name_scope("dropout"):
                    h = tf.nn.dropout(h, self.dropout_keep_prob)
                h_size = filter_size


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[filter_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(h, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
