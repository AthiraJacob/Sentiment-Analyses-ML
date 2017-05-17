#! /usr/bin/env python
'''
One hot vectors with MLP

'''

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_mlp import TextMLP
from tensorflow.contrib import learn
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = 0


# import gensim

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("data_folder", "CR-data/", "Data folder.")

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# pdb.set_trace()

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_train,y_train, x_dev, y_dev = data_helpers.load_data_and_labels(FLAGS.data_folder)

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



filter_sizes_list = [ "1000,500", "500, 500", "1000,1000", "2000,1000", "2000,1000, 500",  "2000,2000, 2000", "1000,1000,1000", "500,500, 500"]
keep_prob_list = [ 0.25, 0.5,  0.7, 1]

# # Training
# # ==================================================
log = open("log.txt","w")
for filter_sizes in filter_sizes_list:
    for dropout_keep_prob in keep_prob_list:

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)


            sess = tf.Session(config=session_conf)
            with sess.as_default():


                cnn = TextMLP(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    filter_sizes=list(map(int, filter_sizes.split(","))),
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout_keep_prob: dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    losses = []
                    accuracies = []
                    batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    for batch in batches_dev:
                        x_batch, y_batch = zip(*batch)
                        feed_dict = {
                          cnn.input_x: x_batch,
                          cnn.input_y: y_batch,
                          cnn.dropout_keep_prob: 1.0
                        }
                        step, summaries, loss, accuracy = sess.run(
                            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        losses.append(loss)
                        accuracies.append(accuracy)
                        print("----- {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    losses = np.array(losses)
                    accuracies = np.array(accuracies)
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, np.mean(losses), np.mean(accuracies)))
                    if writer:
                        writer.add_summary(summaries, step)
                    return np.mean(accuracies)

                # Generate batches
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                best_val = 0
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        val = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        if val>best_val:
                            best_val = val
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                print("Best Validation accuracy: " + str(best_val))
                print("Layers: ({}), Dropout {:g}, Validation acc: {:g}".format(filter_sizes, dropout_keep_prob, best_val), file = log)


        

