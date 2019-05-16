import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split

sampling_rate = 60
features_length = 0
sequence_length = 0
model_path = "\\tmp\\model.ckpt"
source_dir = "output_files\\"


parser = argparse.ArgumentParser()
parser.add_argument("fn", help="Specify the short file name to load; e.g. for file 'All-HC2-6MinWalk.mat' or "
                               "jointangleHC2-6MinWalk.mat, enter 'HC2'. Specify 'all' for all the files available "
                               "in the default directory of the specified file type. Enter anything for 'dc' file type.")
parser.add_argument("--split_size", type=float, nargs="?", const=1,
                    help="Option to split a 'basic' .csv file (i.e. direct copy from JA .mat to .csv) into multiple parts "
                         "for training purposes, where '--unit_size' is the number of seconds to correspond to each"
                         "file label (i.e. '--unit_size'=1 sets 60 frames (due to sampling rate) to each label. "
                         "Defaults to size of complete file (e.g. 360 seconds) for each basic .csv.")
args = parser.parse_args()

file_names = []
x_data, y_data = [], []


if args.fn.lower() != "all":
    if any(args.fn in s for s in os.listdir(source_dir)):
        file_names.append([s for s in os.listdir(source_dir) if args.fn in s][0])
    else:
        print("Cannot find '" + str(args.fn) + "' in '" + source_dir + " '")
        sys.exit()
else:
    file_names = [fn for fn in os.listdir(source_dir)]

for file_name in file_names:
    print("Extracting '" + file_name + " to x_data and y_data....")
    data = pd.read_csv(source_dir + file_name).values
    if not args.split_size:
        args.split_size = 10.0
    y_label = 1 if file_name.split("_")[1][0] == "D" else 0
    sequence_length = int(args.split_size*sampling_rate)
    num_data_splits = int(len(data) / sequence_length)
    for i in range(num_data_splits):
        x_data.append(data[(i * sequence_length):((i + 1) * sequence_length)])
        y_data.append(y_label)


features_length = len(x_data[0][0])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True)


def create_batch_generator(x, y=None, batch_size=64):
    n_batches =len(x)//batch_size
    x = x[:n_batches*batch_size]
    if y is not None:
        y = y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        if y is not None:
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        else:
            yield x[ii:ii+batch_size]


class SentimentRNN(object):
    def __init__(self, features_length, seq_len, lstm_size=256, num_layers=1, batch_size=64,
                 learning_rate=0.0001, embed_size=200):
        self.features_length = features_length
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        tf_x = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.features_length), name='tf_x')
        tf_y = tf.placeholder(tf.float32, shape=(self.batch_size), name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')

        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(
            self.lstm_size), output_keep_prob=tf_keepprob) for i in range(self.num_layers)])
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, tf_x, initial_state = self.initial_state)

        logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=1, activation=None, name='logits')
        logits = tf.squeeze(logits, name='logits_squeezed')
        y_proba = tf.nn.sigmoid(logits, name='probabilities')
        predictions = {'probabilities': y_proba, 'labels': tf.cast(tf.round(y_proba), tf.int32, name='labels')}
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=logits), name='cost')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')

    def train(self, x_train, y_train, num_epochs):
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            iteration = 1
            for epoch in range(num_epochs):
                state = sess.run(self.initial_state)
                for batch_x, batch_y in create_batch_generator(x_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'tf_keepprob:0': 0.5, self.initial_state: state}
                    loss, _, state = sess.run(['cost:0', 'train_op', self.final_state], feed_dict=feed)
                    if iteration % 20 == 0:
                        print("Epoch: %d/%d Iteration: %d | Train loss: %.5f" % (epoch+1, num_epochs, iteration, loss))
                    iteration += 1
            self.saver.save(sess, model_path)

    def predict(self, x_test, return_proba=False):
        preds = []
        with tf.Session(graph = self.g) as sess:
            self.saver.restore(sess, model_path)
            test_state = sess.run(self.initial_state)
            for ii, batch_x in enumerate(create_batch_generator(x_test, None, batch_size=self.batch_size), 1):
                feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0, self.initial_state: test_state}
                if return_proba:
                    pred, test_state = sess.run(['probabilities:0', self.final_state], feed_dict=feed)
                else:
                    pred, test_state = sess.run(['labels:0', self.final_state], feed_dict=feed)
                preds.append(pred)
        return np.concatenate(preds)




rnn = SentimentRNN(features_length=features_length, seq_len=sequence_length, embed_size=256, lstm_size=128,
                   num_layers=3, batch_size=64, learning_rate=0.001)

rnn.train(x_train, y_train, num_epochs=5)
preds = rnn.predict(x_test)
y_true = y_test[:len(preds)]
accuracy = round(((np.sum(preds == y_true) / len(y_true)) * 100), 2)
print("Test Accuracy = " + str(accuracy) + "%")