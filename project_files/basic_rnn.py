import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Does not print to display the many warnings that TensorFlow throws up (many about updating to next version or
#deprecated functionality)
tf.logging.set_verbosity(tf.logging.ERROR)

#Arguments needed here only include the name of the .csv file to use as data for the RNN model and an optional
#'--split_size' argument that splits a basic .csv file into multiple parts (a la in 'matfiles_analysis.py')
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specify the sub-directory with the source directory to search for the file. Must be "
                                "one of 'AD', 'JA', or 'DC'.")
parser.add_argument("fn", help="Specify the short file name to load; e.g. for file 'All-HC2-6MinWalk.mat' or "
                               "jointangleHC2-6MinWalk.mat, enter 'HC2'. Specify 'allfiles' for all the files available "
                               "in the default directory of the specified file type. Enter anything for 'dc' file type.")
parser.add_argument("--split_size", type=float, nargs="?", const=1,
                    help="Option to split a 'basic' .csv file (i.e. direct copy from JA .mat to .csv) into multiple parts "
                         "for training purposes, where '--split_size' is the number of seconds to correspond to each"
                         "file label (i.e. '--split_size'=1 sets 60 frames (due to sampling rate) to each label). "
                         "Defaults to size of complete file (e.g. 360 seconds) for each basic .csv.")
parser.add_argument("--nss", type=bool, nargs="?", const=True,
                    help="Specify if training the network on north star scores instead of binary classification "
                         "of 'D' or 'HC' samples.")
args = parser.parse_args()

#If no optional argument given for '--split_size', defaults to split size = 10, i.e. defaults to splitting files into
#10 second increments
if not args.split_size:
    args.split_size = 10.0

#Location at which to store the created model
model_path = "\\tmp\\model.ckpt"

#Note: CHANGE THIS to location of the source data for the RNN to use
source_dir = "output_files\\direct_csv\\"
sub_dirs = ["AD", "JA", "DC"]

#Note: CHANGE THESE to location of the 3 sub-directories' encompassing directory local to the user that's needed to
#map to the .csvs containing the NSAA information
nsaa_table_path = "C:\\msc_project_files\\"

"""RNN hyperparameters"""
sampling_rate = 60
#Defines the number of sequences that correspond to one 'x' sample (e.g. for split size of 5 seconds and a sampling
#rate of 60Hz, sequence length is 300)
sequence_length = int(args.split_size*sampling_rate)
batch_size = 64
num_lstm_cells = 128
num_rnn_hidden_layers = 2
learn_rate = 0.0001
num_epochs = 20


def preprocessing(source_dir):
    """
        :return given a short file name for 'fn' command-line argument, finds the relevant file in 'source_dir' and
        adds it to 'file_names' (or set 'file_names' to all file names in 'source_dir'), reads in each file name in
        'file_names' as .csv's into dataframes, create corresponding 'y-labels' (with one 'y-label' corresponding to
        each row in the .csv, with it being a 1 if it's from a file beginning with 'D' or 0 otherwise), and shuffling/
        splitting them into training and testing x- and y-components
    """
    file_names = []
    x_data, y_data = [], []

    if args.dir.upper() in sub_dirs:
        source_dir += args.dir.upper() + "\\"
    else:
        print("First arg ('dir') must be one of 'AD', 'JA', and 'DC'.")
        sys.exit()
    if args.fn.lower() != "allfiles":
        if any(args.fn in s for s in os.listdir(source_dir)):
            file_names.append([s for s in os.listdir(source_dir) if args.fn in s][0])
        else:
            print("Cannot find '" + str(args.fn) + "' in '" + source_dir + " '")
            sys.exit()
    else:
        file_names = [fn for fn in os.listdir(source_dir)]
    for file_name in file_names:
        print("Extracting '" + file_name + "' to x_data and y_data....")
        data = pd.read_csv(source_dir + file_name).values
        if args.nss:
            data = add_nsaa_scores(data).values
            y_label = data[0, 0]
        else:
            y_label = 1 if file_name.split("_")[1][0] == "D" else 0
        num_data_splits = int(len(data) / sequence_length)
        for i in range(num_data_splits):
            x_data.append(data[(i * sequence_length):((i + 1) * sequence_length), 19:])
            y_data.append(y_label)

    print(np.shape(x_data))
    print(np.shape(y_data))
    return train_test_split(x_data, y_data, shuffle=True, test_size=0.2)



def add_nsaa_scores(file_df):
    #To make sure that accepted parameter is as a DataFrame
    file_df = pd.DataFrame(file_df)
    mw_tab = pd.read_excel(nsaa_table_path + "6MW-matFiles\\6mw_matfiles.xlsx")
    mw_cols = mw_tab[["ID", "NSAA"]]
    mw_dict = dict(pd.Series(mw_cols.NSAA.values, index=mw_cols.ID).to_dict())

    nsaa_matfiles_tab = pd.read_excel(nsaa_table_path + "NSAA\\matfiles\\nsaa_matfiles.xlsx")
    nsaa_matfiles_cols = nsaa_matfiles_tab[["ID", "NSAA"]]
    nsaa_matfiles_dict = dict(pd.Series(nsaa_matfiles_cols.NSAA.values, index=nsaa_matfiles_cols.ID).to_dict())

    mw_dict.update(nsaa_matfiles_dict)
    nss = [mw_dict[i] for i in [j.split("_")[0] for j in file_df.iloc[:, 0].values]]
    file_df.insert(loc=0, column="NSS", value=nss)

    nsaa_acts_tab = pd.read_excel(nsaa_table_path + "NSAA\\KineDMD data updates Feb 2019.xlsx")
    nsaa_acts_file_names = nsaa_acts_tab.iloc[2:20, 0].values
    nsaa_acts = nsaa_acts_tab.iloc[2:20, 53:70].values
    nsaa_acts_dict = dict(zip(nsaa_acts_file_names, nsaa_acts))
    nsaa_labels = nsaa_acts_tab.iloc[1, 53:70].values

    label_sample_map = []
    for i in range(len(nsaa_labels)):
        inner = []
        for j in range(len(file_df.index)):
            fn = file_df.iloc[j, 1].split("_")[0]
            if fn in nsaa_acts_dict:
                inner.append(nsaa_acts_dict[fn][i])
            else:
                #If patient isn't found in the 'KineDMB' table, assume its a healthy control patient and thus all
                #scores for all activities are perfect (i.e. '2').
                inner.append(2)
        label_sample_map.append(inner)
    for i in range(len(nsaa_labels)):
        file_df.insert(loc=(i+1), column=nsaa_labels[i], value=label_sample_map[i])

    return file_df



def create_batch_generator(x, y=None, batch_size=64):
    n_batches =len(x)//batch_size
    #Ensures 'x' is a multiple of batch size
    x = x[:n_batches*batch_size]
    if y is not None:
        #Ensures 'y' is a multiple of batch size
        y = y[:n_batches*batch_size]
    #For each 'batch_size' chuck of x, yield the next part of the 'x' and 'y' data (i.e. the next 'batch size' samples)
    for ii in range(0, len(x), batch_size):
        if y is not None:
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        else:
            yield x[ii:ii+batch_size]



class BasicRNN(object):
    def __init__(self, features_length, seq_len, lstm_size, num_layers, batch_size, learning_rate):
        """
        :param sets the hyperparameters as 'BasicRNN' object attributes, builds the RNN graph, and initializes
        the global variables
        """
        self.features_length = features_length
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        """
        :return: no return, but sets up the complete RNN architecture to be called upon initialization
        """
        #Placeholders to hold the data that is fed into the RNN (where each batch has shape 'seq_len' x
        # 'features_length' for 'x' data and a single '1' or '0' for 'y' data)
        tf_x = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.features_length), name='tf_x')
        tf_y = tf.placeholder(tf.float32, shape=(self.batch_size), name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')

        #Defines several hidden RNN layers, with 'self.num_layers' layers, 'self.lstm_size' number of cells per
        #layer, and each implementing dropout functionality
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(
            self.lstm_size), output_keep_prob=tf_keepprob) for i in range(self.num_layers)])
        #Sets the initial state for the RNN layers
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        #With the RNN layer architecture defined in 'cells', sets up the layers to feed from input 'x' placeholder
        #into 'cells' and with the above defined 'initial_state'
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, tf_x, initial_state = self.initial_state)


        #Defines the cost based on the sigmoid cross entropy with the RNN output and the 'y' labels, along with
        #the Adam optimizer for the optimizer of choice with a learning rate set by 'self.learning_rate'
        logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=1, activation=None, name='logits')
        logits = tf.squeeze(logits, name='logits_squeezed')
        if args.nss:
            cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf_y, predictions=logits), name='cost')
            predictions = {'cost': cost}
        else:
            # Adds an output layer that feed from the final values emitted from the 'cells' layers with a single neuron
            # to classify for a binary value
            y_proba = tf.nn.sigmoid(logits, name='probabilities')
            predictions = {'probabilities': y_proba, 'labels': tf.cast(tf.round(y_proba), tf.int32, name='labels')}
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=logits), name='cost')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')


    def train(self, x_train, y_train, num_epochs):
        """
        :param the training data for both 'x' and 'y' data is provided, along with the number of training epochs
        that the training will run for
        :return no return, but for each epoch, fetches the next batch of data from the 'x' and 'y' training sets with
        a dropout probability and initial state, and feeds it into the RNN for training following the architecture
        and hyperparameters of 'build()', while printing the loss at every 20 iterations for a given epoch
        """
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            iteration = 1
            for epoch in range(num_epochs):
                print("\n")
                state = sess.run(self.initial_state)
                for batch_x, batch_y in create_batch_generator(x_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'tf_keepprob:0': 0.5, self.initial_state: state}
                    loss, _, state = sess.run(['cost:0', 'train_op', self.final_state], feed_dict=feed)
                    if iteration % 20 == 0:
                        print("Epoch: %d/%d Iteration: %d | Train loss: %.5f" % (epoch+1, num_epochs, iteration, loss))
                    iteration += 1
            self.saver.save(sess, model_path)
        print("\n\n")


    def predict(self, x_test, return_proba=False, regress=False):
        """
            :param feeds in the 'x' testing data with which we wish to compute the predicted values (1 or 0) for
            each of the 2-dimensional samples (where each sample is 'self.seq_length' x 'self.features_length'), with
            an option to return the labels rather than the probabilities for assessment purposes
            :return: list of predicted values that are the result of feeding through the 'x' testing data through the
            now-trained RNN model
        """
        preds = []
        with tf.Session(graph = self.g) as sess:
            self.saver.restore(sess, model_path)
            test_state = sess.run(self.initial_state)
            for ii, batch_x in enumerate(create_batch_generator(x_test, None, batch_size=self.batch_size), 1):
                feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0, self.initial_state: test_state}
                if return_proba:
                    pred, test_state = sess.run(['probabilities:0', self.final_state], feed_dict=feed)
                elif regress:
                    pred, test_state = sess.run(['logits_squeezed:0', self.final_state], feed_dict=feed)
                else:
                    pred, test_state = sess.run(['labels:0', self.final_state], feed_dict=feed)
                preds.append(pred)
        return np.concatenate(preds)


#Extracts the training and testing data, builds the RNN based on the hyperparameters initially set, and trains the model
x_train, x_test, y_train, y_test = preprocessing(source_dir)
rnn = BasicRNN(features_length=len(x_train[0][0]), seq_len=sequence_length, lstm_size=num_lstm_cells,
                   num_layers=num_rnn_hidden_layers, batch_size=batch_size, learning_rate=learn_rate)

rnn.train(x_train, y_train, num_epochs=num_epochs)

preds = rnn.predict(x_test, regress=args.nss)
#Ensures the true 'y' values are the same length and the predicted values (so 'preds' and 'y_true' have the same shape)
y_true = y_test[:len(preds)]


if args.nss:
    mse = mean_squared_error(y_true=y_true, y_pred=preds)
    print("\n\nMean Squared Error = " + str(round(mse, 4)))
    mae = mean_absolute_error(y_true=y_true, y_pred=preds)
    print("Mean Absolute Error = " + str(round(mae, 4)))
else:
    #Calculates and prints the accuracy to the user
    accuracy = round(((np.sum(preds == y_true) / len(y_true)) * 100), 2)
    print("\n\nTest Accuracy = " + str(accuracy) + "%")