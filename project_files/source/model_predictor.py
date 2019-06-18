import argparse
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter


"""Section below encompasses the three arguments that are required to be passed to the script. This arguments ensures 
that the right file is loaded in to be tested with the pre-trained models, along with the arguments themselves informing 
the script about which pre-trained model to select to test the file on."""
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory the prediction file is contained in so as to process "
                                "the file accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles' or 'NSAA'.")
parser.add_argument("ft", help="Specify type of .mat file that the .csv is to come from, being one of 'JA' (joint "
                               "angle), 'AD' (all data), or 'DC' (data cube). Alternatively, supply a name of a "
                               "measurement (e.g. 'position', 'velocity', 'jointAngle', etc.) if the file is to be "
                               "trained on raw measurements.")
parser.add_argument("fn", help="Specify the short file name of a .csv to be the predictor from 'source_dir'; e.g. for file "
                               "'All_D2_stats_features.csv', enter 'D2'.")
args = parser.parse_args()



#Note: CHANGE THIS to location of the 3 sub-directories' encompassing directory local to the user.
local_dir = "C:\\msc_project_files\\"

#Other locations and sub-dir nameswithin 'local_dir' that will contain the files we need, as dictated by assuming the
#user has previously run the required scripts (e.g. 'comp_stat_vals', 'ext_raw_measures', etc.)
source_dir = local_dir + "output_files\\"
output_dir = local_dir + "output_files\\RNN_outputs\\"
model_dir = local_dir + "output_files\\rnn_models\\"
sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\", "direct_csv\\"]
sub_sub_dirs = ["AD\\", "JA\\", "DC\\"]
raw_measurements = ["position", "velocity", "acceleration", "angularVelocity", "angularAcceleration",
                "sensorFreeAcceleration", "sensorMagneticField", "jointAngle", "jointAngleXZY"]

#Appends the sub_dir name to 'source_dir' if it's one of the allowed names
if args.dir + "\\" in sub_dirs:
    source_dir += args.dir.upper() + "\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'direct_csv'.")
    sys.exit()

#Appends the file type name to 'source_dir' if it's one of the allowed types or, if it's a raw measurement name, change
#the 'source_dir' name completely to account for a new location within 'local_dir' that contains the raw measurements
if args.ft + "\\" in sub_sub_dirs:
    source_dir += args.ft + "\\"
elif args.dir == "NSAA" and args.ft in raw_measurements:
    source_dir = local_dir + "NSAA\\matfiles\\" + args.ft + "\\"
else:
    print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
          "\'JA', or \'DC\' (unless dir is give as 'NSAA', where 'ft' can be a measurement name).")
    sys.exit()

#Makes sure that the 'fn' of the file we wish to test on (e.g. 'D11') is the name of a file within 'source_dir' and,
#if it is, select this as 'file_name'
if any(args.fn == s.upper().split("-")[0] for s in os.listdir(source_dir)):
    file_name = [s for s in os.listdir(source_dir) if args.fn in s][0]
else:
    print("Cannot find '" + str(args.fn) + "' in '" + source_dir + "'")
    sys.exit()

#Contains the complete path to the file we are loading
full_file_name = source_dir + file_name

#Loads the 'RNN Results.xlsx' file, finds the row that corresponds to data on 'dir' and 'ft' arguments that were given,
#finds the cell on that row that contains info about the sequence length for the relevant model corresponding to this
#setup, and extracts and stores this value in 'sequence_length'; this ensures that the data we feed in to test the model
#is the same shape as it is expecting
results_xlsx = pd.read_excel("..\\RNN Results.xlsx")
rnn_args = [m for m in results_xlsx.iloc[:, 4].values if str(m) != "nan"]
rnn_command = [m for m in rnn_args if args.dir == m.split(" ")[2] and args.ft == m.split(" ")[3]][0]
sequence_length = int(results_xlsx.iloc[rnn_args.index(rnn_command), 9].split(" = ")[1])

#Loads the .csv data from the given file name and divides it up into a 3D format based on 'sequence_length'; for
#example, if 'df_x' is of shape (1000, 50) and 'sequence_length' = 60, then 'x_data' becomes (16, 60, 50). Note that
#the 'leftover' data at the end of 'df_x' that does not fit into a (60, 50) shape is discarded.
df_x = pd.read_csv(full_file_name, index_col=0).values
x_data = [df_x[sequence_length*i:(sequence_length*(i+1)), :] for i in range(int(len(df_x)/sequence_length))]

#Loads the file that contains information on each subject and their corresponding overall and individual NSAA scores,
#and three 'true' labels for the file are extracted: 'y_label_dhc' gets 1 if the short name of the file (e.g. 'D11')
#begins with a 'D', else it gets a 0; 'y_label_overall' gets an integer value between 0 and 34 of the overall NSAA score,
#and 'y_label_acts' gets the 17 individiual NSAA acts as a list as scores between 0 and 2
df_y = pd.read_excel("..\\documentation\\nsaa_6mw_matfiles.xlsx")
if args.dir != "direct_csv" and not source_dir.startswith(local_dir):
    y_label_dhc = 1 if file_name.split("_")[2][0] == "D" else 0
elif source_dir.startswith(local_dir):
    y_label_dhc = 1 if file_name.split("_")[0][0] == "D" else 0
else:
    y_label_dhc = 1 if file_name.split("_")[1][0] == "D" else 0
y_index = df_y.index[df_y["ID"] == args.fn]
y_label_overall = df_y.loc[y_index, "NSAA"].values[0]
y_label_acts = [int(num) for num in df_y.iloc[y_index, 5:].values[0]]


#The models that are to be tested by the script are selected based on their names; e.g. if 'dir'=NSAA and 'ft'=position,
#then 'NSAA_position_all_acts', 'NSAA_position_all_dhc', and 'NSAA_position_all_overall' will be loaded as directory
#names containing the trained models
models = [m for m in os.listdir(model_dir) if m.startswith(args.dir) and not m.endswith("indiv")
          and args.ft == m.split("_")[1]]



def create_batch_generator(x, y=None, batch_size=64):
    """
    :param 'x', which is the data to create batches out of and, if 'y' is supplied as well, create batches out of
    this as well based on the 'batch_size' provided
    :return: the next batch of data to the calling function
    """
    n_batches = len(x)//batch_size
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



#Stores the strings to write AFTER all the models have run so all results are printed at the end, rather than some being
#printed and then obscured by the model setup text
output_strs = []
for model in models:
    #Contains one of 'acts', 'indiv', or 'dhc'
    output_type = model.split("_")[-1]
    #Complete model path to the directory of the model in question
    model_path = model_dir + model
    preds = []
    with tf.Session(graph=tf.Graph()) as sess:
        #Loads the model and restores from it's final checkpoint
        new_saver = tf.train.import_meta_graph(model_path + "\\model.ckpt.meta", clear_devices=True)
        new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
        #For each batch of the data from 'full_file_name', feed it through the trained model to get predictions based
        #on the type of model (e.g. '1's or '0's if output_type == "dhc", ints between 0 and 34 if output_type ==
        #"overall", and lists of 17 values between 0 and 1 if output_type == "dhc") and appends these to 'preds'
        for ii, batch_x in enumerate(create_batch_generator(x_data, None, batch_size=64), 1):
            feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0}
            if output_type == "overall":
                preds.append(sess.run('logits_squeezed:0', feed_dict=feed))
            else:
                preds.append(sess.run('labels:0', feed_dict=feed))
    #Flatten these values to a 1D list
    preds = np.concatenate(preds)
    #Based on what type the model is, add output lines to 'output_strs' that gives info on the true 'y' label of the
    #file (e.g. true 'D' or 'HC' label, true overall NSAA score, or true single acts scores) and what the model predicted
    if output_type == "overall":
        output_strs.append(str("True 'Overall NSAA Score' = " + str(y_label_overall)))
        output_strs.append(str("Predicted 'Overall NSAA Score' = " + str(round(float(np.mean(preds)), 2))))
    elif output_type == "dhc":
        true_label = "D" if y_label_dhc == 1 else "HC"
        pred_label = "D" if max(set(list(preds)), key=list(preds).count) == 1 else "HC"
        d_percen = np.round((np.sum(preds)/len(preds)), 4)
        output_strs.append(str("True 'D/HC Label' = " + true_label))
        output_strs.append(str("Predicted 'D/HC Label' = " + pred_label))
        output_strs.append(str("Percentage of predicted 'D' sequences = " + str(d_percen*100) + "%"))
        output_strs.append(str("Percentage of predicted 'HC' sequences = " + str((1-d_percen)*100) + "%"))
    else:
        preds = np.transpose(preds)
        pred_acts = [Counter(preds[i]).most_common()[0][0] for i in range(len(preds))]
        output_strs.append(str("True 'Acts Sequence' = " + str(y_label_acts)))
        output_strs.append(str("Predicted 'Acts Sequence' = " + str(pred_acts)))
    output_strs.append("")

#Prints to the user all the output lines that were generated by testing on all the models
print("\n")
for output_str in output_strs:
    print(output_str)
