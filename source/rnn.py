import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import StandardScaler
import pyexcel as pe
from matplotlib import pyplot as plt
from math import floor
import data_balancer
from settings import local_dir, source_dir, output_dir, sub_dirs, sub_sub_dirs, raw_measurements, batch_size, \
    nsaa_6mw_path, model_shapes_path
from ft_sel_red import ft_red_select



#Does not print to display the many warnings that TensorFlow throws up (many about updating to next version or
#deprecated functionality)
tf.logging.set_verbosity(tf.logging.ERROR)

#Necessary for imported variable to work as global
source_dir = source_dir

"""Section below encompasses all the arguments that are required to setup and train the model. This includes the 
name of the directory to source the files from to train the model, the type of files that this directory contains, 
the name(s) of the file to train the model, and the choice of training model (i.e. what the model should be setup 
to predict as output). Option arguments include those to specify the sequence length the RNN should deal with, the 
percentage of sequence overlap between train/test samples, the number of epochs to use, and whether to write the 
settings to the output results file (these go to default values if not specified)."""
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory to use so as to process the files contained within "
                                "them accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles' or 'NSAA'. "
                                "Alternatively, specify 'cnn_data' if building models from the data used for 'cnn' project.")
parser.add_argument("ft", help="Specify type of .mat file that the .csv is to come from, being one of 'JA' (joint "
                               "angle), 'AD' (all data), or 'DC' (data cube). Alternatively, supply a name of a "
                               "measurement (e.g. 'position', 'velocity', 'jointAngle', etc.) if the file is to be "
                               "trained on raw measurements. Additionally, if user wishes to concatenate file types "
                               "along the 'features' dimension, specify multiple file types separated by commas.")
parser.add_argument("fn", help="Specify the short file name of a .csv to load from 'source_dir'; e.g. for file "
                               "'All_D2_stats_features.csv', enter 'D2'. Specify 'all' for all the files available "
                               "in the 'source_dir'.")
parser.add_argument("choice", help="Specify the choice of what the network should be training towards. Specify 'dhc' "
                                   "for simple binary classification of sequences, 'overall' for single-target "
                                   "regression to predict the overall north star scores for sequences, 'acts' for "
                                   "multi-target classification of possible NSAA activities that have taken place "
                                   "in the sequence and what their corresponding individual NSAA would be, or 'indiv' "
                                   "for predicting a score of 0, 1, or 2 from 'single-act' source files.")
parser.add_argument("--seq_len", type=float, nargs="?", const=1,
                    help="Option to split a source file (be it a raw joint-angle file or a JA/DC/AD stats features file) "
                         "into multiple parts for training purposes, where '--seq_len' is the number of 'rows' of data "
                         "to correspond to each file label/score(s).")
parser.add_argument("--seq_overlap", type=float, nargs="?", const=1,
                    help="Option to specify the proportion of overlap in divided up sequences. E.g. set to '0.75' to "
                         "have each of e.g. 742 sequences to overlap 75% of each other, resulting in 927 new sequences.")
parser.add_argument("--discard_prop", type=float, nargs="?", const=1,
                    help="Option to discard every nth element to reduce the size of the sequence length while keeping "
                         "context high (e.g. if seq_len=180 and discard_prop=0.2, then it discards every 5th element of "
                         "the sequences and is left with a sequence length of 144 but with the context window of the "
                         "original sequence length of 180).")
parser.add_argument("--write_settings", type=bool, nargs="?", const=True, default=False,
                    help="Option to write the hyperparameters of the RNN directly to the RNN results .csv file in "
                         "the appropriate row.")
parser.add_argument("--create_graph", type=bool, nargs="?", const=True, default=False,
                    help="Option to create a graph of the true values against the predicted values and save them "
                         "to the 'Graphs' subdirectory within the 'documentation' directory.")
parser.add_argument("--epochs", type=int, nargs="?", const=1,
                    help="Option to set number of epochs to use in this run of the model.")
parser.add_argument("--other_dir", type=str, nargs="?", const=True, default=False,
                    help="Option to include an additional source directory as part of the data to train the model on. "
                         "Must not be the same as 'dir' and must be one '6minwalk-matfiles', 6MW-matFiles', or 'NSAA'.")
parser.add_argument("--leave_out", type=str, nargs="?", const=True, default=False,
                    help="Option to specify the short file names of the file to leave out of the train and test set "
                         "altogether, and thus use it reliably in 'model_predictor.py'. Note that it removes ALL files "
                         "with a matching short name, so '--leave_out=D4' would exclude 'D4', 'd4' and 'D4v2' files. "
                         "Alternatively, specify short files names so particular files are left out (e.g. "
                         "'--leave_out=HC8-008' would leave 'HC8-009' in the data set. Split multiple files to be left "
                         "out by commas.")
parser.add_argument("--balance", type=str, nargs="?", const=True, default=False,
                    help="Option to balance the data based on overall NSAA scores. Set as 'up' to upsample samples with "
                         "overall NSAA y-labels to that of the most frequent value or 'down' to downsample to that of "
                         "the least frequent value.")
parser.add_argument("--use_frc", type=bool, nargs="?", const=True, default=False,
                    help="Option to use the 'FRC_' files for 'AD' files instead of 'FR_' files, i.e. use files where "
                         "dimensionality reduction has been applied the same way over all files rather than on a "
                         "file-by-file basis.")
parser.add_argument("--use_sac", type=bool, nargs="?", const=True, default=False,
                    help="Option to use the concatenated files for 'single-act' files over all activities per subject "
                         "instead of separate files for each activity.")
parser.add_argument("--standardize", type=bool, nargs="?", const=True, default=False,
                    help="Option to standardize all the data (i.e. so each feature column of the total data set "
                         "(following splitting into sequences) is N(0, 1).")
parser.add_argument("--noise", type=float, nargs="?", const=True, default=False,
                    help="Option to add Gaussian-distributed noise to each of the features. Specify a proportion of "
                         "N(0, 1) noise to the add to the data set.")
parser.add_argument("--batch", type=bool, nargs="?", const=True, default=False,
                    help="Option that is only set if the script is run from a batch file to access the external files "
                         "in a correct way.")
parser.add_argument("--pca", type=str, nargs="?", const=True, default=False,
                    help="Specify if wishing reduce the feature size of the data set training the model to the number "
                         "of features specified by the value passed to this argument.")
parser.add_argument("--balance_allmatfiles", type=int, nargs="?", const=True, default=False,
                    help="Specify this to only use a smaller sample of the overall 'allmatfiles' data set, as the "
                         "base contains ~8x as many files as NSAA. Value given is the maximum number of files per "
                         "subject that we can draw from the 'allmatfiles' directory.")
parser.add_argument("--balance_nmb", type=int, nargs="?", const=True, default=False,
                    help="Specify this to only use a smaller sample of the overall 'nmb' data set, as the "
                         "base contains ~8x as many files as NSAA. Value given is the maximum number of files per "
                         "subject that we can draw from the 'allmatfiles' directory.")
parser.add_argument("--no_testset", type=bool, nargs="?", const=True, default=False,
                    help="Specify this to set the train/test ratio to 0 so to use all the available data as training. "
                         "Note that this results in no results being written to console or the 'RNN Results' directory.")
parser.add_argument("--leave_out_version", type=str, nargs="?", const=True, default=False,
                    help="Specify this with the name of a version of subjects (e.g. 'V2') to leave out of the files "
                         "used to train the models. Use this if wish to train models on one version of subjecst to "
                         "then evaluate model generalisation to other versions of already-seen subjects.")
args = parser.parse_args()

#If no optional argument given for '--seq_len', defaults to seq_len = 10, i.e. defaults to splitting files into
#10 length increments
if not args.seq_len:
    args.seq_len = 10.0

#If no optional argument given for '--seq_overlap', defaults to seq_overlap = 0, i.e. defaults to no overlap of sequences
if not args.seq_overlap:
    args.seq_overlap = 0

#Ensures the sequence overlap isn't '1', which would result in an infinite number of sequences as the sequence window
#would never 'move along' the data
if args.seq_overlap == 1:
    print("Can't set overlap to be 100% of each sequence...")
    sys.exit()

#If no optional argument given for '--discard_prop', defaults to discard_prop = 0, i.e. defaults to not discarding
#any parts of the sequences
if not args.discard_prop:
    args.discard_prop = 0

#Ensures that, if '--balance' is set, it is either 'up' to upsample the data or 'down' to downsample the data
if args.balance:
    if not args.balance == "up" and not args.balance == "down":
        print("Optional arg ('--balance') must be set to either 'up' or 'down'.")
        sys.exit()


#Location at which to store the created model that will be used by 'model_predictor.py'
model_path = local_dir + "output_files\\rnn_models\\" + '_'.join(sys.argv[1:]) + "\\model.ckpt"


#Available choices of model outputs for 'choice' argument to take and available measurement names for 'ft' to take 
#if not one of 'sub_sub_dirs'
choices = ["dhc", "overall", "acts", "indiv"]
choice = None

#Global variable to hold strings that should be printed at the end of 'rnn' running if '--balance' is set
balance_strs = []

"""RNN hyperparameters"""
x_shape = None
y_shape = None
sampling_rate = 60
#Defines the number of sequences that correspond to one 'x' sample
sequence_length = int(args.seq_len)
num_lstm_cells = 128
num_rnn_hidden_layers = 2
learn_rate = 0.001
num_epochs = 20
test_ratio = 0.2
num_acts = 17

#Only sets 'num_epochs' to different value if '--epochs' optional argument is provided
if args.epochs:
    num_epochs = int(args.epochs)

#Only sets the 'test_ratio' to 0 (i.e. use all available data for model training) if '--no_testset' optional
#argument is provided
if args.no_testset:
    test_ratio = 0

#If '--other_dir' is set, make sure it's a permitted name before continuing
if args.other_dir:
    if not args.other_dir + "\\" in sub_dirs or args.other_dir == args.dir:
        print("Optional arg '--other_dir' must be one of '6minwalk-matfiles', '6MW-matFiles', 'NSAA', "
              "'NMB', or 'direct_csv' and must not be the same name as 'dir'....")
        sys.exit()

#Sets the paths for external files based on the whether the script was called from a batch file or not
nsaa_6mw_path = "..\\" + nsaa_6mw_path if args.batch else nsaa_6mw_path
model_shapes_path = "..\\" + model_shapes_path if args.batch else model_shapes_path



def preprocessing(dir, ft):
    """
        :return given a short file name for 'fn' command-line argument, finds the relevant file in 'source_dir' and
        adds it to 'file_names' (or set 'file_names' to all file names in 'source_dir'), reads in each file name in
        'file_names' as .csv's into DataFrames, create corresponding 'y-labels' (with one 'y-label' corresponding to
        each row in the .csv, with it being a 1 if it's from a file beginning with 'D' or 0 otherwise), and shuffling/
        splitting them into training and testing x- and y-components
    """
    file_names = []
    x_data, y_data = [], []
    #Declare 'sequence_length' as global to be able to modify later on if '--discard_prop' is set
    global sequence_length
    #Appends the sub_dir name to 'source_dir' if it's one of the allowed names
    global source_dir
    #Sets 'y_value_balance' and 'y_data_balance' to be used in function scope if '--balance' is set
    y_label_balance, y_data_balance = None, []
    #If 'dir'=cnn, call the 'cnn_preprocessing' function instead and return the x/y train/test splits that the
    #function returns
    if dir == "cnn_data":
        return cnn_preprocessing()

    if dir + "\\" in sub_dirs:
        source_dir += dir + "\\"
    else:
        print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
              "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'NMB', or 'direct_csv'.")
        sys.exit()
    #Change 'source_dir' to point to the directory of raw measurement files, single-act raw measurements files,
    #or another type of file (e.g. 'AD' or 'JA')
    if dir == "NSAA" and args.choice == "indiv" and ft in raw_measurements:
        source_dir = local_dir + "NSAA\\matfiles\\act_files\\" + ft + "\\"
    elif dir == "NSAA" and ft in raw_measurements and args.use_sac:
        source_dir = local_dir + "NSAA\\matfiles\\act_files_concat\\" + ft + "\\"
    elif dir == "NSAA" and args.choice == "indiv" and ft + "\\" in sub_sub_dirs:
        source_dir += ft + "\\act_files\\"
    elif dir == "NSAA" and ft + "\\" in sub_sub_dirs and args.use_sac:
        source_dir += ft + "\\act_files_concat\\"
    elif ft + "\\" in sub_sub_dirs and dir != "6MW-matFiles":
        source_dir += ft + "\\"
    elif dir == "allmatfiles" and ft in raw_measurements:
        source_dir = local_dir + "allmatfiles\\" + ft + "\\"
    elif dir == "NSAA" and ft in raw_measurements:
        source_dir = local_dir + "NSAA\\matfiles\\" + ft + "\\"
    elif dir == "6minwalk-matfiles":
        if ft == "AD" or ft == "JA":
            source_dir += + ft + "\\"
        elif ft in raw_measurements:
            source_dir = local_dir + "6minwalk-matfiles\\all_data_mat_files\\" + ft + "\\"
        else:
            print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
                  "\'JA', or \'DC\' (unless dir is give as 'NSAA', where 'ft' can be a measurement name).")
            sys.exit()
    elif dir == "6MW-matFiles" or dir == "NMB":
        if ft == "AD":
            source_dir = local_dir + "\\output_files\\" + dir + "\\AD\\"
        elif ft in raw_measurements:
            source_dir = local_dir + dir + "\\" + ft + "\\"
        else:
            print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
                  "\'JA', or \'DC\' (unless dir is give as 'NSAA', where 'ft' can be a measurement name).")
            sys.exit()
    else:
        print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
              "\'JA', or \'DC\' (unless dir is give as 'NSAA', where 'ft' can be a measurement name).")
        sys.exit()

    #Appends to the list of files the single file name corresponding to the 'fn' argument or all available files within
    #'source_dir' if 'fn' is 'all'
    if args.fn.lower() != "all":
        if any(args.fn == s.upper().split("-")[0] for s in os.listdir(source_dir)):
            file_names.append([s for s in os.listdir(source_dir) if args.fn in s][0])
        else:
            print("Cannot find '" + str(args.fn) + "' in '" + source_dir + "'")
            sys.exit()
    else:
        try:
            file_names = [fn for fn in os.listdir(source_dir)]
        except FileNotFoundError:
            print(dir, "not a valid 'dir' name for when choice='" + args.choice + "'...")
            sys.exit()

    #Sets 'choice' equal to the 'choice' argument if it's one of the allowed output RNN choices (i.e. 'dhc', 'acts',
    #'indiv', or 'overall')
    global choice
    if args.choice in choices:
        choice = args.choice
    else:
        print("Must provide a choice of 'dhc' for simple binary classification of sequences, 'overall' for single-target "
              "regression to predict the overall north star scores for sequences, 'acts' for multi-target "
              "classification of possible NSAA activities that have taken place in the sequence and what their "
              "corresponding individual NSAA would be, or 'indiv' for predicting a score of 0, 1, or 2 from "
              "'single-act' source files.")
        sys.exit()

    #Ensures that only the written files from feature select/reduct script are used if present (i.e. if the directory
    #we are concerned with is not within 'direct_csv')
    if dir != "direct_csv" and source_dir.startswith(local_dir + "output_files\\"):
        if not args.use_frc:
            file_names = [fn for fn in file_names if fn.startswith("FR_")]
        else:
            file_names = [fn for fn in file_names if fn.startswith("FRC_")]
    elif ft == "AD":
        if not args.use_frc:
            file_names = [fn for fn in file_names if fn.startswith("FR_")]
        else:
            file_names = [fn for fn in file_names if fn.startswith("FRC_")]

    if not file_names:
        print("No files found in given directory for given args...")
        sys.exit()

    #Removes any file that contains in the name the optional argument '--leave_out', split by commas
    if args.leave_out:
        lo_names = args.leave_out.split(",")
        for lo_n in lo_names:
            file_names = [fn for fn in file_names if lo_n not in fn]

    #Removes any file that contains in the name the optional argument '--leave_out_version'
    if args.leave_out_version:
        file_names = [fn for fn in file_names if args.leave_out_version not in fn]

    #If the '--balance_allmatfiles' optional argument is set, rebalance the 'allmatfiles' data set to use a reduced
    #number of files
    if dir == "allmatfiles" and args.balance_allmatfiles:
        print("Initial number of files within 'allmatfiles': " + str(len(file_names)))
        #Ensures that the number of files selected for each subject name in 'allmatfiles' are randomly chosen for a
        #given subject, as opposed to the first occuring files
        np.random.seed(42)
        np.random.shuffle(file_names)

        #Setups a dictionary to count the number of files we have selected per subject and an empty list to hold the
        #selected file names
        fn_counts = {fn.split("jointangle")[1].split("-")[0]:0 for fn in file_names}
        new_file_names = []

        #For each file name, adds it to the new list of file names to include if we haven't reached the maximum number
        #allowed ('--balance_allmatfiles') for that subject
        for fn in file_names:
            short_fn = fn.split("jointangle")[1].split("-")[0]
            if fn_counts[short_fn] < args.balance_allmatfiles:
                new_file_names.append(fn)
                fn_counts[short_fn] += 1

        #Sets the newly chosen file names to the list of files from which to build the model
        file_names = new_file_names
        print("Reduced number of files within 'allmatfiles' to use to train models (max = " +
              str(args.balance_allmatfiles) + " per subject) : " + str(len(file_names)))
    elif dir == "NMB" and args.balance_nmb:
        print("Initial number of files within 'nmb': " + str(len(file_names)))
        #Ensures that the number of files selected for each subject name in 'nmb' are randomly chosen for a
        #given subject, as opposed to the first occuring files
        np.random.seed(42)
        np.random.shuffle(file_names)

        #Setups a dictionary to count the number of files we have selected per subject and an empty list to hold the
        #selected file names
        fn_counts = {fn.split("-")[0]: 0 for fn in file_names}
        new_file_names = []

        #For each file name, adds it to the new list of file names to include if we haven't reached the maximum number
        #allowed ('--balance_nmb') for that subject
        for fn in file_names:
            short_fn = fn.split("-")[0]
            if fn_counts[short_fn] < args.balance_nmb:
                new_file_names.append(fn)
                fn_counts[short_fn] += 1

        # Sets the newly chosen file names to the list of files from which to build the model
        file_names = new_file_names
        print("Reduced number of files within 'NMB' to use to train models (max = " +
              str(args.balance_nmb) + " per subject) : " + str(len(file_names)))

    #For each file name that we are dealing with (all files names in 'source_dir' if 'fn' is 'all, else a single
    #file name), adds 'y' labels based on what type of model output we are training for and divide up both 'x' and 'y'
    #data into sequences
    for file_name in file_names:
        print("Extracting '" + file_name + "' to x_data and y_data....")
        #Read in the data from the corresponding .csv
        data = pd.read_csv(source_dir + file_name)

        #If the model output type is 'overall', add the NSAA scores if they aren't included already,
        #and get the overall NSAA score from the first row's first cell in the file as the 'y_label'
        if choice == "overall":
            if data.columns.values[1] != "NSS":
                try:
                    data = add_nsaa_scores(data.values)
                except KeyError:
                    print(file_name + " not found as entry in either 'nsaa_6mw_info', skipping...")
                    continue
            data = data.values
            if dir != "direct_csv" and source_dir.startswith(local_dir + "output_files\\"):
                y_label = data[0, 1]
            else:
                y_label = data[0, 0]
        #If the model output type is 'dhc', set 'y_label' to 1 if the first letter of the short file name within the
        #file name is 'D', else set it to 0 (i.e. if it's a 'HC' file)
        elif choice == "dhc":
            data = data.values
            if dir != "direct_csv" and not source_dir.startswith(local_dir):
                y_label = 1 if file_name.split("_")[2][0].upper() == "D" else 0
            elif ft == "AD":
                y_label = 1 if file_name.split("_")[2][0].upper() == "D" else 0
            elif file_name.startswith("All"):
                y_label = 1 if file_name.split("_")[0].split("-")[1][0].upper() == "D" else 0
            elif source_dir.startswith(local_dir) and ft != "AD":
                y_label = 1 if file_name.split("_")[0][0].upper() == "D" else 0
            else:
                y_label = 1 if file_name.split("_")[1][0].upper() == "D" else 0
        #If the model output type is 'acts', add the NSAA scores if they aren't included already, and get the
        #individual NSAA activity scores from 17 cells in the first row of the data
        elif choice == "acts":
            if data.columns.values[1] != "NSS":
                try:
                    data = add_nsaa_scores(data.values)
                except KeyError:
                    print(file_name + " not found as entry in either '6mw_matfiles.xlsx', 'nsaa_matfiles.xlsx', "
                                           "or 'KineDMD data updates Feb 2019.xlsx', 'skipping...")
                    continue
            data = data.values
            if dir != "direct_csv" and not source_dir.startswith(local_dir):
                y_label = data[0][2:19]
            elif ft == "AD":
                y_label = data[0][2:19]
            else:
                y_label = data[0][1:18]
        #If the model output type is 'indiv', add the NSAA scores if they aren't included already, get the name of the
        #activity the file is sourced from (e.g. 'act5') and, based on this, retrieve the value of the correct column
        #from the first row of data (e.g. retrieve the 5th column of the first row if file contains 'act5')
        else:
            if data.columns.values[1] != "NSS":
                try:
                    data = add_nsaa_scores(data.values)
                except KeyError:
                    print(file_name + " not found as entry in either '6mw_matfiles.xlsx', 'nsaa_matfiles.xlsx', "
                                           "or 'KineDMD data updates Feb 2019.xlsx', 'skipping...")
                    continue
            data = data.values
            y_label = data[0][int(file_name.split(".")[0].split("_")[1].split("act")[1])]

        #If the '--balance' optional argument is set, then set the 'y_label_balance' to overall NSAA score for the file,
        #regardless of the 'choice' arg, to use as a means to balance the data
        if args.balance:
           y_label_balance = data_balancer.ext_label_dist(file_name=file_name, batch=args.batch)

        #Determine the number of data splits needed based on the size of the data file and the desired sequence length,
        #including rounding it down and accounting for the sequence overlap proportion)
        num_data_splits = int(len(data) / sequence_length)
        num_data_splits = int(num_data_splits * (1/(1-args.seq_overlap)))
        start, end = 0, 0
        #For each desired sequence, determine the start and end positions of the sequence in 'data', extract this from
        #the body of 'data', append this to 'x_data', and append the previously-determined 'y_label' to 'y_data'
        for i in range(num_data_splits):
            end = start + sequence_length
            #Prevents moving window from 'clipping' the end of the data rows and getting a number of rows of
            #less than 'sequence_length'
            if end > len(data):
                continue
            #Discards every 'nth' row of a sequence if the user sets the optional '--discard_prop' argument to reduce
            #the sequence size while keeping the original context window the same
            split_data = data[start:end]
            if args.discard_prop:
                if args.discard_prop == 0:
                    print("Cannot set '--discard_prop to '0' as would give a zero division error when trying to "
                          "take the 'nth' element...")
                    sys.exit()
                if args.discard_prop <= 0.5:
                    split_data = np.asarray([row for i, row in enumerate(split_data)
                                             if i % floor(1/args.discard_prop) != 0])
                else:
                    split_data = np.asarray([row for i, row in enumerate(split_data)
                                             if i % floor(1/round((1-args.discard_prop), 5)) == 0])

            if dir == "direct_csv" and choice == "dhc":
                x_data.append(split_data[:, 1:])
            elif dir == "direct_csv":
                x_data.append(split_data[:, 19:])
            elif source_dir.startswith(local_dir) and choice == "dhc" and not ft == "AD":
                x_data.append(split_data[:, 1:])
            elif source_dir.startswith(local_dir) and "output_files" not in source_dir:
                x_data.append(split_data[:, 19:])
            else:
                x_data.append(split_data[:, 21:])
            y_data.append(y_label)
            #If we wish to balance the data, append the overall NSAA score to a separate list used to balance the data
            if args.balance:
                y_data_balance.append(y_label_balance)
            #Note: overlap lengths are rounded DOWN via 'int'
            start += int(sequence_length*(1-args.seq_overlap))

    global x_shape
    global y_shape
    x_shape = np.shape(x_data)
    y_shape = np.shape(y_data)
    print("X shape =", x_shape)
    print("Y shape =", y_shape)
    #Sets the global 'sequence_length' if '--discard_prop' is called to ensure RNN is setup with the correct
    #placeholder variable shape
    if args.discard_prop:
        sequence_length = len(x_data[0])

    #Rebalances the data based on the value of 'args.balance'
    if args.balance:
        if args.balance == "up":
            x_data, y_data, balance_s = data_balancer.upsample(x_data, y_data, y_data_balance)
        else:
            x_data, y_data, balance_s = data_balancer.downsample(x_data, y_data, y_data_balance)
        #Sets the global variable to print out balance strings at a later point
        global balance_strs
        balance_strs = balance_s
        x_shape = np.shape(x_data)
        y_shape = np.shape(y_data)
        print("Balanced X shape =", x_shape)
        print("Balanced Y shape =", y_shape)

    #If the '--standardize' optional argument is given, reshape the 'x' data into a 2D array, standardize
    #each of the features, and reshape it back into its original shape (note: np.concatenate and np.reshape aren't used
    #in order to ensure the data is written back as we would expect with the same number of lines)
    if args.standardize:
        x_data = [sample for sequence in x_data for sample in sequence]
        print("Standardizing the data set...")
        x_data = StandardScaler().fit_transform(x_data)
        x_data = [[x_data[i*x_shape[1] + j] for j in range(x_shape[1])] for i in range(x_shape[0])]
    #If the '--noise' argument is given, add Gaussian noise to every feature of every sample of every sequence
    #in the data set
    if args.noise:
        x_data = [sample for sequence in x_data for sample in sequence]
        print("Adding Gaussian noise to the data set...")
        x_data += np.random.normal(0, np.std(np.array(x_data, dtype=np.float64), axis=0) * args.noise, np.shape(x_data))
        x_data = [[x_data[i * x_shape[1] + j] for j in range(x_shape[1])] for i in range(x_shape[0])]

    #Appends the arguments that were used to invoke the model and its sequence length to a file that stores
    #the sequence lengths (to be used by the 'model_predictor.py' script
    model_shape = pd.read_excel(model_shapes_path)
    new_model_shape = model_shape.append(
        {"dir": dir, "ft": ft, "measure": args.choice, "seq_len": x_shape[1]}, ignore_index=True)
    new_model_shape.to_excel(model_shapes_path, index=False)

    return x_data, y_data



def cnn_preprocessing():
    """
        :return:given the name for a 'left-out' file (i.e. the subject to be used as a testing set) via the '--left_out'
        optional argument (required here), the relevant training and testing .csv files (one for each) are loaded in
        and sequences are extracted from this and given corresponding 'y' labels based on the 'choice' arg, followed by
        shuffling these sets are passing them along to the calling 'preprocessing' function, which terminates with the
        returned data from this function
    """

    global sequence_length
    cnn_data_dir = local_dir + "cnn_data\\"

    train_test_data = [[[], []], [[], []]]
    if any(dir for dir in os.listdir(cnn_data_dir) if dir.startswith(args.leave_out)):
        leave_out_dir = [dir for dir in os.listdir(local_dir + "cnn_data\\") if dir.startswith(args.leave_out)][0]
    else:
        print("Must provide '--leave_out' arg as name of subject to leave out of training...")
        sys.exit()

    global choice
    if args.choice == "dhc":
        choice = "dhc"
    elif args.choice == "overall":
        choice = "overall"
    else:
        print("For 'dir'=cnn, the 'choice' arg must be one of 'overall' or 'dhc'...")
        sys.exit()



    for i, t_t in enumerate(("train", "test")):
        file_name = cnn_data_dir + leave_out_dir + "\\" + t_t + ".csv"
        file_data = pd.read_csv(file_name)
        print("\n" + t_t.title() + " file '" + file_name + "' of shape: " + str(np.shape(file_data)))
        print("Extracting '" + file_name + "' data...")
        x_data = file_data.iloc[:, 1:29].values
        if choice == "dhc":
            y_data = file_data.iloc[:, 29].values
        else:
            y_data = file_data.iloc[:, 30].values

        #Determine the number of data splits needed based on the size of the data file and the desired sequence length,
        #including rounding it down and accounting for the sequence overlap proportion)
        num_data_splits = int(len(x_data) / sequence_length)
        num_data_splits = int(num_data_splits * (1 / (1 - args.seq_overlap)))

        print("Splitting '" + file_name + "' into sequences....")
        start, end = 0, 0
        # For each desired sequence, determine the start and end positions of the sequence in 'data', extract this from
        # the body of 'data', append this to 'x_data', and append the previously-determined 'y_label' to 'y_data'
        for j in range(num_data_splits):
            end = start + sequence_length
            # Prevents moving window from 'clipping' the end of the data rows and getting a number of rows of
            # less than 'sequence_length'
            if end > len(x_data):
                continue
            # Discards every 'nth' row of a sequence if the user sets the optional '--discard_prop' argument to reduce
            # the sequence size while keeping the original context window the same
            split_data = x_data[start:end]
            if args.discard_prop:
                if args.discard_prop == 0:
                    print("Cannot set '--discard_prop to '0' as would give a zero division error when trying to "
                          "take the 'nth' element...")
                    sys.exit()
                if args.discard_prop <= 0.5:
                    split_data = np.asarray([row for j, row in enumerate(split_data)
                                             if j % floor(1 / args.discard_prop) != 0])
                else:
                    split_data = np.asarray([row for j, row in enumerate(split_data)
                                             if j % floor(1 / round((1 - args.discard_prop), 5)) == 0])

            train_test_data[i][0].append(split_data)
            #Appends the label that corresponds to the label of the middle row of the sequence
            train_test_data[i][1].append(y_data[int((end + start)/2)])
            # Note: overlap lengths are rounded DOWN via 'int'
            start += int(sequence_length * (1 - args.seq_overlap))


    global x_shape
    global y_shape
    x_shape = np.shape(np.concatenate((train_test_data[0][0], train_test_data[1][0]), axis=0))
    y_shape = np.shape(np.concatenate((train_test_data[0][1], train_test_data[1][1]), axis=0))

    print("\n")
    print("X train shape =", np.shape(train_test_data[0][0]))
    print("Y train shape =", np.shape(train_test_data[0][1]))
    print("X test shape =", np.shape(train_test_data[1][0]))
    print("Y test shape =", np.shape(train_test_data[1][1]))

    # Rebalances the data based on the value of 'args.balance'
    if args.balance:
        if args.balance == "up":
            train_test_data[0][0], train_test_data[0][1], balance_s = \
                data_balancer.upsample(train_test_data[0][0], train_test_data[0][1], train_test_data[0][1])
        else:
            train_test_data[0][0], train_test_data[0][1], balance_s = \
                data_balancer.downsample(train_test_data[0][0], train_test_data[0][1], train_test_data[0][1])
        # Sets the global variable to print out balance strings at a later point
        global balance_strs
        balance_strs = balance_s
        x_shape = np.shape(train_test_data[0][0])
        y_shape = np.shape(train_test_data[0][1])
        print("Balanced X train shape =", x_shape)
        print("Balanced Y train shape =", y_shape)

    # Sets the global 'sequence_length' if '--discard_prop' is called to ensure RNN is setup with the correct
    # placeholder variable shape
    if args.discard_prop:
        sequence_length = len(train_test_data[0][0][0])

    #Shuffles the training and testing data sets, as it isn't automatically done in this 'preprocessing' function
    #due to not using the 'train_test_split' function
    z = list(zip(train_test_data[0][0], train_test_data[0][1]))
    np.random.shuffle(z)
    x_train, y_train = zip(*z)
    z = list(zip(train_test_data[1][0], train_test_data[1][1]))
    np.random.shuffle(z)
    x_test, y_test = zip(*z)

    return np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)



def add_nsaa_scores(file_df):
    """
    :param 'file_df', which contains the values in a 2D numpy array, to have the values NSAA scores appended on each
    of its rows
    :return: the same data as before, but with the overall and individual NSAA scores appended at the beginning of
    each row of the data
    """
    #To make sure that accepted parameter is as a DataFrame
    file_df = pd.DataFrame(file_df)

    #For the table of data that we have on the subjects, load in the table, find the columns with ID and
    #overall NSAA scores, and create a dictionary of matching values, e.g. {'D4': 15, 'D11: 28,...}, with all values
    #from each table
    nsaa_6mw_tab = pd.read_excel(nsaa_6mw_path)
    nsaa_6mw_cols = nsaa_6mw_tab[["ID", "NSAA"]]
    nsaa_overall_dict = dict(pd.Series(nsaa_6mw_cols.NSAA.values, index=nsaa_6mw_cols.ID).to_dict())

    #If the first column's names begins with "jointangle", remove this part
    if file_df.iloc[0, 0].startswith("jointangle"):
        for i, row in file_df.iterrows():
            row[0] = row[0].split("jointangle")[1]

    #Adds column of overall NSAA scores at position 0 of every row of the data values, with the NSAA score being
    #appended determined by the short file name of the data as found at the beginning of each row of the data
    nss = [nsaa_overall_dict[i.upper()[:-2] if i.upper().endswith("V2") else i.upper()]
           for i in [j.split("_")[0] for j in file_df.iloc[:, 0].values]]
    file_df.insert(loc=0, column="NSS", value=nss)

    #Loads the data that contains information about single act NSAA scores from the .xlsx file, extracts the
    #file names and single-acts columns, and creates a list of label names (i.e. the names of the activities) and a
    #dictionary that maps the label names to a list of single-act scores
    nsaa_single_dict = {}
    for name, acts in zip(nsaa_6mw_tab.loc[:, "ID"].values, nsaa_6mw_tab.iloc[:, 5:].values):
        if not any(np.isnan(acts)):
            nsaa_single_dict[name] = acts
    nsaa_act_labels = nsaa_6mw_tab.columns.values[5:]

    #For each label name and for every row, adds the score that is found in the single-acts dictionary for the relevant
    #activity for a given short file name (if it isn't found in the dictionary, add a '2' as we're assuming it's a
    #healthy control patient), add these together, and insert each new row of values at the beginning of the old rows
    #so each now have the additional single-act scores and overall NSAA scores at the beginning of each row and return it
    label_sample_map = []
    for i in range(len(nsaa_act_labels)):
        inner = []
        for j in range(len(file_df.index)):
            fn = file_df.iloc[j, 1].split("_")[0].upper()
            fn = fn[:-2] if fn.endswith("V2") else fn
            if fn in nsaa_single_dict:
                inner.append(nsaa_single_dict[fn][i])
            elif fn.startswith("HC"):
                inner.append(2)
            else:
                #If patient isn't found in the table (and thus we don't have info on the individual NSAA scores),
                #don't continue with the file and move onto the next one
                raise KeyError
        label_sample_map.append(inner)
    for i in range(len(nsaa_act_labels)):
        file_df.insert(loc=(i+1), column=nsaa_act_labels[i], value=label_sample_map[i])
    return file_df



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



def write_to_csv(trues, preds, output_strs, open_file=False):
    """
    :param 'trues', which are a list of 'true' values for each of the test sequences that have been tested on the model,
    'preds', which are the predicted values corresponding to each 'true' value, and 'output_strs' which are the strings
    that have been printed to the console and will also be stored in the output file (with 'open_file' being whether to
    immediately open it upon writing to it
    :return: no return, but instead writes the sequence numbers, true values, predicted values, console output, and
    model settings to a new file within 'output_dir'
    """
    print("\nWriting true and predicted values to .csv....")
    df = pd.DataFrame()
    df["Sequence Number"] = np.arange(1, len(trues)+1)
    if choice != "acts":
        df["Predictions"] = preds
        df["Trues"] = trues
    else:
        df["Predictions"] = ["[" + " ".join(str(num) for num in preds[i]) + "]" for i in range(len(preds))]
        df["Trues"] = ["[" + " ".join(str(num) for num in trues[i]) + "]" for i in range(len(trues))]
    df["Results"] = ""
    df.iloc[0, -1] = ' '.join(sys.argv[1:])
    df.iloc[1, -1] = ", ".join(output_strs)
    settings = ["X shape = " + str(x_shape), "Y shape = " + str(y_shape), "Test ratio = " + str(test_ratio),
                "Sequence length = " + str(sequence_length), "Features length = " + str(len(x_train[0][0])),
                "Num epochs = " + str(num_epochs), "Num LSTM units per layer = " + str(num_lstm_cells),
                "Num hidden layers = " + str(num_rnn_hidden_layers), "Learning rate = " + str(learn_rate)]
    df["Settings"] = pd.Series(settings)

    #Create 'output_dir' if it doesn't exist and, if it does exist, remove the existing file and write the data frame
    #to this file, opening it afterwards if required; if the '--write_settings' argument is given, append the settings
    #written to this file to the 'RNN Results.ods' file directly rather than manually copying them over
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_index = 1
    full_output_name = output_dir + '_'.join(sys.argv[1:]) + "_RNN_trues_preds" + str(file_index) + ".csv"
    while os.path.exists(full_output_name):
        file_index += 1
        full_output_name = "_".join(full_output_name.split("_")[:-1]) + "_" + str(file_index) + ".csv"
    df.to_csv(full_output_name, index=False, float_format="%.2f")
    if open_file:
        os.startfile(full_output_name)
    if args.write_settings:
        settings = ["" for i in range(7)] + settings
        sheet = pe.get_sheet(file_name="..\\RNN Results.ods")
        sheet.row += settings
        sheet.save_as("..\\RNN Results.ods")
    if args.create_graph:
        fig, ax = plt.subplots()
        ax.scatter(trues, preds, alpha=0.10)
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x)
        plt.title("Plot of true overall NSAA scores against predicted overall NSAA scores")
        plt.xlabel("True overall NSAA scores")
        plt.ylabel("Predicted overall NSAA scores")
        file_name = args.dir + "_" + args.ft + "_" + args.choice + "_trues_preds"
        plt.savefig("..\\documentation\\Graphs\\" + file_name)
        plt.gcf().set_size_inches(10, 10)
        plt.show()



class RNN(object):
    def __init__(self, features_length, seq_len, lstm_size, num_layers, batch_size, learning_rate, num_acts):
        """
        :param sets the hyperparameters as 'RNN' object attributes, builds the RNN graph, and initializes
        the global variables
        """
        self.features_length = features_length
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_acts = num_acts

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
        if choice != "acts":
            tf_y = tf.placeholder(tf.float32, shape=(self.batch_size), name='tf_y')
        else:
            tf_y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_acts), name='tf_y')
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
        if choice != "acts":
            logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=1, activation=None, name='logits')
            logits = tf.squeeze(logits, name='logits_squeezed')
        else:
            logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=self.num_acts, activation=None, name='logits')

        if choice == "overall" or choice == "indiv":
            cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf_y, predictions=logits), name='cost')
            predictions = {'cost': cost}
        elif choice == "dhc":
            # Adds an output layer that feed from the final values emitted from the 'cells' layers with a single neuron
            # to classify for a binary value
            y_proba = tf.nn.sigmoid(logits, name='probabilities')
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=logits), name='cost')
            predictions = {'probabilities': y_proba, 'labels': tf.cast(tf.round(y_proba), tf.int32, name='labels')}
        else:
            cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf_y, predictions=logits), name='cost')
            predictions = {'labels': tf.cast(tf.round(logits), tf.int32, name='labels'), 'cost': cost}
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
                state = sess.run(self.initial_state)
                for batch_x, batch_y in create_batch_generator(x_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'tf_keepprob:0': 0.5, self.initial_state: state}
                    loss, _, state = sess.run(['cost:0', 'train_op', self.final_state], feed_dict=feed)
                    if iteration % 20 == 0:
                        print("Epoch: %d/%d Iteration: %d | Train loss: %.5f" % (epoch+1, num_epochs, iteration, loss))
                    iteration += 1
            self.saver.save(sess, model_path)
        print("\n\n")



    def predict(self, x_test, return_proba=False):
        """
            :param feeds in the 'x' testing data with which we wish to compute the predicted values (1 or 0) for
            each of the 2-dimensional samples (where each sample is 'self.seq_length' x 'self.features_length'), with
            an option to return the labels rather than the probabilities for assessment purposes
            :return: list of predicted values that are the result of feeding through the 'x' testing data through the
            now-trained RNN model
        """
        preds = []
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess, model_path)
            test_state = sess.run(self.initial_state)
            for ii, batch_x in enumerate(create_batch_generator(x_test, None, batch_size=self.batch_size), 1):
                feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0, self.initial_state: test_state}
                if return_proba:
                    pred, test_state = sess.run(['probabilities:0', self.final_state], feed_dict=feed)
                elif choice == "overall" or choice == "indiv":
                    pred, test_state = sess.run(['logits_squeezed:0', self.final_state], feed_dict=feed)
                else:
                    pred, test_state = sess.run(['labels:0', self.final_state], feed_dict=feed)
                preds.append(pred)
        return np.concatenate(preds)



#Sets the default sequence length source dir so the originals can be retrieved later after having been modified by
#different 'ft's or by 'altdirs'
original_seq_len = sequence_length
original_source_dir = source_dir

#Separates the 'ft' arg into list of file types, separated by a comma
fts = [ft for ft in args.ft.split(",")]
#Sets the scope of the variables to store all data that is concatenated together to global scope and a variable to
#check whether any data has been entered into this variable yet
x_data, y_data = None, None
data_entered = False


#For each of the directories that we wish to use as a source of the file data...
for dir in args.dir.split(","):
    #For each of the file types of the argument (separated by commas), preprocesses the data from all files for this
    #file type, concatenates it with any previous data from other file types along the features dimension (i.e. horizontal
    #concatenation (while checking that it is possible to do so, assuming the first 2 dimensions are the same)
    fts_x_data, fts_y_data = None, None
    fts_data_entered = False
    for ft in fts:
        sequence_length = original_seq_len
        source_dir = original_source_dir
        #Extracts the training and testing data
        ft_x_data, ft_y_data = preprocessing(dir, ft)
        #if ft != "AD" or x_data is None:
        fts_x_data = np.concatenate((fts_x_data, ft_x_data), axis=2) if fts_data_entered else ft_x_data
        fts_y_data = fts_y_data if fts_y_data else ft_y_data
        fts_data_entered = True
        """
        else:
            print("'AD' is not a valid 'dir' choice when concatenating file types due to different sized "
                             "sequence lengths and numbers of samples; not including 'AD'...")
            continue
        """
    x_data = np.concatenate((x_data, fts_x_data), axis=0) if data_entered else fts_x_data
    y_data = np.concatenate((y_data, fts_y_data), axis=0) if data_entered else fts_y_data
    data_entered = True

print("Final concatenated 'X' data shape =", np.shape(x_data))
print("Final concatenated 'Y' data shape =", np.shape(y_data))

#Repeats the preprocessing for 'other_dir' if specified by setting 'dir' to the value of 'other_dir' and then setting
#it back to the original value after extracting the data from 'other_dir' and adding it to the 'dir' data
if args.other_dir:
    old_dir = args.dir
    args.dir = args.other_dir
    sequence_length = original_seq_len
    alt_x_data, alt_y_data = preprocessing(args.other_dir, args.ft)
    x_data = np.concatenate((x_data, alt_x_data), axis=0)
    y_data = np.concatenate((y_data, alt_y_data), axis=0)
    args.dir = old_dir


#If the optional arg '--pca' is given, reduces the dimensionality of the data set to the dimensions specified
#by the argument via calling the relevant function from 'ft_sel_red.py'
if args.pca:
    try:
        pca = int(args.pca)
        print("Reducing the dimensions of data via 'PCA'....")
        x_data = [sample for sequence in x_data for sample in sequence]
        x_data, y_data = ft_red_select(x_data, y_data, "pca", False, False, pca)
        x_data = [[x_data[i * x_shape[1] + j] for j in range(x_shape[1])] for i in range(x_shape[0])]
        print("Reduced-dim X Shape = " + str(np.shape(x_data)))
        print("Reduced-dim Y Shape = " + str(np.shape(y_data)))
    except ValueError:
        if args.pca != "all":
            print("Optional arg '--pca' must contain int value or 'all' to keep all features...")
            sys.exit()



print("Splitting the data into training and testing components...")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, test_size=test_ratio)



#Builds the RNN based on the hyperparameters initially set, and trains the model
rnn = RNN(features_length=len(x_train[0][0]), seq_len=sequence_length, lstm_size=num_lstm_cells,
                   num_layers=num_rnn_hidden_layers, batch_size=batch_size, learning_rate=learn_rate, num_acts=num_acts)

rnn.train(x_train, y_train, num_epochs=num_epochs)

#If the optional arg '--no_testset' is set, finish the the program after completing the training of the model
if args.no_testset:
    print("Model built: finishing without testing as '--no_testset' is set...")
    sys.exit()

preds = rnn.predict(x_test)
#Ensures the true 'y' values are the same length and the predicted values (so 'preds' and 'y_true' have the same shape)
y_true = y_test[:len(preds)]


#Based on what type of model was created (i.e. with which output type), create several strings based on various
#evaluation metrics (i.e. the performance results of the model in question on test data) and append them to a list,
#which is then printed to the console and also passed to the 'write_to_csv' so they can be written to file
output_strs = []
if choice == "overall" or choice == "indiv":
    output_strs.append("Mean Squared Error = " + str(round(mean_squared_error(y_true=y_true, y_pred=preds), 4)))
    output_strs.append("Mean Absolute Error = " + str(round(mean_absolute_error(y_true=y_true, y_pred=preds), 4)))
    output_strs.append("Root Mean Squared Error = " + str(round(np.sqrt(mean_squared_error(y_true=y_true, y_pred=preds)), 4)))
    output_strs.append("R^2 Score = " + str(round(r2_score(y_true=y_true, y_pred=preds), 4)))
elif choice == "dhc":
    output_strs.append(str(cm(y_true=y_true, y_pred=preds)) + "\n(Note: row names = true vals, col names = pred vals;"
                       + "cm[0,0] = True HC, cm[0,1] = False D, cm[1,0] = False HC, c[1,1] = True D)")
    output_strs.append("Test Accuracy = " + str(round(((np.sum(preds == y_true) / len(y_true)) * 100), 2)) + "%")
else:
    ind_sum, all_sum = 0, 0
    for i in range(len(preds)):
        ind_sum += np.sum(preds[i] == y_true[i])
        all_sum = all_sum + 1 if list(preds[i]) == list(y_true[i]) else all_sum
    output_strs.append("Individual Activity Accuracy = " + str(round(((ind_sum / (len(y_true)*num_acts)) * 100), 2)) + "%")
    output_strs.append("All Activities Accuracy = " + str(round(((all_sum / len(preds)) * 100), 2)) + "%")


print("\n\n")
for output_str in output_strs:
    print(output_str)

#Prints to the console the before and after shapes of the data if the '--balance' optional arg is set
if args.balance:
    for balance_str in balance_strs:
        print(balance_str)

write_to_csv(trues=y_true, preds=preds, output_strs=output_strs)