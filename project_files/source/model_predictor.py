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
                               "trained on raw measurements, or multiple 'ft's split by commas.")
parser.add_argument("fn", help="Specify the short file name of a .csv to be the predictor from 'source_dir'; e.g. for file "
                               "'All_D2_stats_features.csv', enter 'D2'.")
args = parser.parse_args()


#Note: CHANGE THIS to location of the 3 sub-directories' encompassing directory local to the user.
local_dir = "C:\\msc_project_files\\"

#Note: must match the batch size of the models that have been trained (default is 64)
batch_size = 64


#Other locations and sub-dir nameswithin 'local_dir' that will contain the files we need, as dictated by assuming the
#user has previously run the required scripts (e.g. 'comp_stat_vals', 'ext_raw_measures', etc.)
source_dir = local_dir + "output_files\\"
output_dir = local_dir + "output_files\\RNN_outputs\\"
model_dir = local_dir + "output_files\\rnn_models\\"
sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\", "direct_csv\\"]
sub_sub_dirs = ["AD\\", "JA\\", "DC\\"]
file_types = ["AD", "position", "velocity", "acceleration", "angularVelocity", "angularAcceleration",
                "sensorFreeAcceleration", "sensorMagneticField", "jointAngle", "jointAngleXZY"]
output_types = ["acts", "dhc", "overall"]

#Appends the sub_dir name to 'source_dir' if it's one of the allowed names
if args.dir + "\\" in sub_dirs:
    source_dir += args.dir.upper() + "\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'direct_csv'.")
    sys.exit()

fts, sds = [], []
for ft in args.ft.split(","):
    fts.append(ft)
    if ft + "\\" in sub_sub_dirs:
        sds.append(source_dir + ft + "\\")
    elif ft in file_types and args.dir == "NSAA":
        sds.append(local_dir + "NSAA\\matfiles\\" + ft + "\\")
    else:
        print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
              "\'JA', or \'DC\' (unless dir is give as 'NSAA', where 'ft' can be a measurement name), and each part "
              "must be separated by a comma for each measurement to use in the ensemble.")
        sys.exit()

fns = []
for sd in sds:
    if any(args.fn == s.upper().split("-")[0] for s in os.listdir(sd)) \
            or any(args.fn == s.upper().split("_")[2] for s in os.listdir(sd)):
        if sd.endswith("AD\\"):
            fns.append([s for s in os.listdir(sd) if args.fn == s.split("_")[1]][0])
        else:
            fns.append([s for s in os.listdir(sd) if args.fn in s][0])
    else:
        print("Cannot find '" + str(args.fn) + "' in '" + sd + "'")
        sys.exit()

#Creates a list of the full file path names to load the models of, given the type of file
full_file_names = []
for sd, fn in zip(sds, fns):
    if fn.startswith("AD"):
        full_file_names.append(sd + "FR_" + fn)
    else:
        full_file_names.append(sd + fn)

#The models that are to be tested by the script are selected based on their names; e.g. if 'dir'=NSAA and 'ft'=position,
#then 'NSAA_position_all_acts', 'NSAA_position_all_dhc', and 'NSAA_position_all_overall' will be loaded as directory
#names containing the trained models
models = []
for ot in output_types:
    inner_models = []
    for i, ft in enumerate(fts):
        if any(fn for fn in os.listdir(model_dir) if fn.split("_")[-1] == ot and fn.split("_")[1] == ft):
            inner_models.append([fn for fn in os.listdir(model_dir) if fn.split("_")[-1] == ot and fn.split("_")[1] == ft][0])
        else:
            inner_models.append(None)
    models.append(inner_models)

#Removes empty 'inner_model' lists (i.e. ones only populated by 'Nones') in the case where all measurements don't have
#a specific output type model
new_models = []
for inner_models in models:
    if not all(not model for model in inner_models):
        new_models.append(inner_models)
models = new_models

#Reads in the .xlsx file with the file shapes in them and extracts the sequence lengths from the most recent entry in
#the tabel that corresponds to the model with a specific source dir, file type, and output type
model_shape = pd.read_excel("..\\documentation\\model_shapes.xlsx")


sequence_lengths = [[model_shape.loc[(model_shape["dir"] == args.dir) & (model_shape["ft"] == model.split("_")[1]) &
                                     (model_shape["measure"] == model.split("_")[-1])].iloc[-1, -1] if model else None
                     for model in inner_models] for inner_models in models]



def preprocessing(full_file_name, sequence_length):
    #Loads the .csv data from the given file name and divides it up into a 3D format based on 'sequence_length'; for
    #example, if 'df_x' is of shape (1000, 50) and 'sequence_length' = 60, then 'x_data' becomes (16, 60, 50). Note that
    #the 'leftover' data at the end of 'df_x' that does not fit into a (60, 50) shape is discarded.
    if not "AD" in full_file_name.split("\\")[-1]:
        df_x = pd.read_csv(full_file_name, index_col=0).values
    else:
        df_x = pd.read_csv(full_file_name, index_col=0).iloc[:, 20:].values

    x_data = [df_x[sequence_length*i:(sequence_length*(i+1)), :] for i in range(int(len(df_x)/sequence_length))]

    #Loads the file that contains information on each subject and their corresponding overall and individual NSAA scores,
    #and three 'true' labels for the file are extracted: 'y_label_dhc' gets 1 if the short name of the file (e.g. 'D11')
    #begins with a 'D', else it gets a 0; 'y_label_overall' gets an integer value between 0 and 34 of the overall NSAA score,
    #and 'y_label_acts' gets the 17 individiual NSAA acts as a list as scores between 0 and 2
    df_y = pd.read_excel("..\\documentation\\nsaa_6mw_info.xlsx")
    if args.dir != "direct_csv" and not source_dir.startswith(local_dir):
        y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[2][0] == "D" else 0
    elif source_dir.startswith(local_dir):
        y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[0][0] == "D" else 0
    else:
        y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[1][0] == "D" else 0
    y_index = df_y.index[df_y["ID"] == args.fn]
    y_label_overall = df_y.loc[y_index, "NSAA"].values[0]
    y_label_acts = [int(num) for num in df_y.iloc[y_index, 5:].values[0]]

    return x_data, y_label_dhc, y_label_overall, y_label_acts



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
for i, inner_models in enumerate(models):
    #'output_type' contains one of 'acts', 'indiv', or 'dhc'`
    output_type = None
    for model in inner_models:
        if model:
            output_type = model.split("_")[-1]
    preds = []
    y_label_dhc, y_label_overall, y_label_acts = (None,)*3
    for j, full_file_name in enumerate(full_file_names):
        #If there is a 'None' sequence length (i.e. a corresponding model for the output type and file type doesn't
        #exist), then skip over it
        if not sequence_lengths[i][j]:
            continue
        x_data, y_label_dhc, y_label_overall, y_label_acts = preprocessing(full_file_name, sequence_lengths[i][j])
        #Complete model path to the directory of the model in question
        model_path = model_dir + inner_models[j]
        inner_preds = []
        with tf.Session(graph=tf.Graph()) as sess:
            #Loads the model and restores from it's final checkpoint
            new_saver = tf.train.import_meta_graph(model_path + "\\model.ckpt.meta", clear_devices=True)
            new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
            #If there aren't enough sequences within the file being tested to make up a full batch (i.e. len(x_data)
            #< batch_size), then replicate it until it's the size of at least batch size
            new_x_data = []
            while len(new_x_data) < batch_size:
                new_x_data += x_data
            x_data = new_x_data
            #For each batch of the data from 'full_file_name', feed it through the trained model to get predictions based
            #on the type of model (e.g. '1's or '0's if output_type == "dhc", ints between 0 and 34 if output_type ==
            #"overall", and lists of 17 values between 0 and 1 if output_type == "dhc") and appends these to 'preds'
            for ii, batch_x in enumerate(create_batch_generator(x_data, None, batch_size=batch_size), 1):
                feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0}
                if output_type == "overall":
                    inner_preds.append(sess.run('logits_squeezed:0', feed_dict=feed))
                else:
                    inner_preds.append(sess.run('labels:0', feed_dict=feed))
        #Flatten these values to a 1D list
        preds.append(np.concatenate(inner_preds))
    #Aggregate results for measurement over
    if output_type == "acts":
        preds = np.transpose(preds, (1, 2, 0))
        preds = [[Counter(elems).most_common()[0][0] for elems in row] for row in preds]
    elif output_type == "dhc":
        preds = np.transpose(preds)
        preds = [Counter(elems).most_common()[0][0] for elems in preds]
    else:
        preds = np.transpose(preds)
        preds = [np.mean(elems) for elems in preds]



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
