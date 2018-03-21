import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from img_data_utils import *
import math
import time


class ShallowModel(object):
    """A shallow model
    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    """
    def __init__(self, reg_strength=1e-3):
        tf.reset_default_graph()
        self.reg_strength = reg_strength

        # Define placeholders
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.num_classes = 2

        out_1 = tf.layers.conv2d(inputs=self.X,
                                 filters=32,
                                 kernel_size=[7, 7],
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
        out_2 = tf.layers.conv2d(inputs=out_1,
                                 filters=128,
                                 kernel_size=[7, 7],
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
        out_3 = tf.layers.max_pooling2d(inputs=out_2, pool_size=[2, 2], strides=2)
        out_4 = tf.layers.batch_normalization(inputs=out_3, axis=3, training=self.is_training)
        out_5 = tf.layers.dense(inputs=tf.reshape(out_4, [-1, 32*32*128]), units=1000, activation=tf.nn.relu)
        out_6 = tf.layers.dropout(inputs=out_5, rate=0.10, training=self.is_training)
        out_7 = tf.layers.dense(inputs=out_6, units=self.num_classes)
        probs = tf.contrib.layers.softmax(logits=out_7)
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, self.num_classes),
                                                                     logits=out_7)

        # Define loss, prediction, and accuracy
        self.mean_loss = tf.reduce_mean(cross_entropy_loss)
        self.correct_prediction = tf.equal(tf.argmax(probs, axis=1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Define optimization objective, a.k.a. train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.op_objective = tf.train.AdagradOptimizer(learning_rate=0).minimize(self.mean_loss)

    def run(self, sess, inputX, inputY, epochs=1, print_every=100, batch_size=100, mode='train'):
        iterations = []
        minibatch_losses = []
        minibatch_accurs = []

        N = inputX.shape[0]
        train_indices = np.arange(N)
        np.random.shuffle(train_indices)

        variables = [self.mean_loss, self.correct_prediction, self.accuracy]
        if mode == 'train':
            variables.append( self.op_objective)

        iter_cnt = 0
        for ep in range(epochs):
            num_correct_per_epoch = 0
            num_iterations = int(math.ceil(N / batch_size))
            for i in range(num_iterations):
                # Generate indices for the batch
                start_idx = (i * batch_size) % N
                idx_range = train_indices[start_idx:start_idx + batch_size]

                feed_dict = {
                    self.X: inputX[idx_range, :],
                    self.y: inputY[idx_range],
                    self.is_training: mode == 'train'
                }

                actual_batch_size = inputY[idx_range].shape[0]

                # Compute loss and number of correct predictions
                if mode == 'train':
                    mean_loss, corr, acc, _ = sess.run(variables, feed_dict=feed_dict)
                else:
                    mean_loss, corr, acc = sess.run(variables, feed_dict=feed_dict)

                minibatch_accurs.append(acc)
                minibatch_losses.append(mean_loss * actual_batch_size)
                iterations.append(iter_cnt)

                num_correct_per_epoch += float(np.sum(corr))

                if mode == 'train' and (iter_cnt % print_every) == 0:
                    mini_batch_acc = float(np.sum(corr)) / float(actual_batch_size)
                    mini_batch_loss = mean_loss * float(actual_batch_size)
                    print "Iteration {0}: with mini-batch loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt, mini_batch_loss, mini_batch_acc)

                iter_cnt += 1

            # End of epoch, that means went over all training examples at least once.
            avg_accuracy = num_correct_per_epoch / N
            avg_loss = np.sum(minibatch_losses) / N
            print "Epoch {0}, overall loss = {1:.3g} and avg training accuracy = {2:.3g}".format(ep+1,
                                                                                                avg_loss,
                                                                                                avg_accuracy)
        return iterations, minibatch_losses, minibatch_accurs


def main():
    model = ShallowModel()
    # data = data_utils.get_preprocessed_CIFAR10('datasets/cifar-10-batches-py', should_transpose=False)
    data = load_jpg_from_dir("datasets/dog-vs-cat-train/", resize_px=64, num_images_per_class=10000)
    Xtr, ytr = data['X'], data['y']

    data = load_jpg_from_dir("datasets/dog-vs-cat-train/", resize_px=64, num_images_per_class=2500, start_idx=10001)
    Xval, yval = data['X'], data['y']

    print "Training data X shape={0}".format(Xtr.shape)
    print "Training data y shape={0}".format(ytr.shape)

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())

            t0 = time.time()
            print 'Start training'
            iters, losses, accuracies = model.run(sess, Xtr, ytr, epochs=1, batch_size=100, print_every=1)
            t1 = time.time()
            print "Elapsed time using GPU: " + str(t1 - t0)
            print accuracies

            plt.grid(True)

            plt.figure(1)
            plt.subplot(2, 1, 1)
            plt.plot(iters, losses)
            plt.title('Mini-batch Losses Over Iterations')
            plt.xlabel('Iteration number')
            plt.ylabel('Mini-batch training loss')

            plt.figure(1)
            plt.subplot(2, 1, 2)
            plt.plot(iters, accuracies)
            plt.title('Mini-batch Training Accuracies Over Iterations')
            plt.xlabel('Iteration number')
            plt.ylabel('Mini-batch training accuracy')
            plt.show()

            print 'Performing validations'
            iters, losses, accuracies = model.run(sess, Xval, yval, epochs=1, batch_size=100, mode='validate')
            print accuracies

if __name__ == '__main__':
    main()
