import argparse
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from matplotlib import pyplot as plt
from settings import local_dir, batch_size, source_dir, output_dir, \
    model_dir, sub_dirs, sub_sub_dirs, file_types, output_types, model_pred_path, model_shapes_path, nsaa_6mw_path
from ft_sel_red import ft_red_select


"""Section below encompasses the three arguments that are required to be passed to the script. This arguments ensures 
that the right file is loaded in to be tested with the pre-trained models, along with the arguments themselves informing 
the script about which pre-trained model to select to test the file on."""
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory the prediction file is contained in so as to process "
                                "the file accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles', "
                                "'NSAA', 'allmatfiles', or 'left-out'.")
parser.add_argument("ft", help="Specify type of .mat file that the .csv is to come from, being one of 'JA' (joint "
                               "angle), 'AD' (all data), or 'DC' (data cube). Alternatively, supply a name of a "
                               "measurement (e.g. 'position', 'velocity', 'jointAngle', etc.) if the file is being "
                               "predicted is a raw measurement, or multiple raw measurements split by commas if the "
                               "file is an 'AD' file and user wishes to test it on raw measurement models.")
parser.add_argument("fn", help="Specify the short file name of a .csv to be the predictor from 'source_dir'; e.g. for file "
                               "'All_D2_stats_features.csv', enter 'D2'.")
parser.add_argument("--alt_dirs", type=str, nargs="?", const=True, default=False,
                    help="Optional argument to use directory types other than 'dir' to predict from(e.g. if "
                         "'dir'=allmatfiles, can specify '--alt_dirs'=NSAA,6minwalk-matfiles to predict the "
                         "'allmatfile' file on models trained on 'NSAA' and '6minwalk-matfiles' files.")
parser.add_argument("--show_graph", type=bool, nargs="?", const=True, default=False,
                    help="Option to show the 'trues-preds' graph that is created and saved by the script.")
parser.add_argument("--handle_dash", type=bool, nargs="?", const=True, default=False,
                    help="Set this to add a dash ('-') to the beginning of the 'fn' arg; primarily used to get the "
                         "correct short name of files within 'allmatfiles', as 'fn' can't start with a dash.")
parser.add_argument("--file_num", type=str, nargs="?", const=True, default=False,
                    help="Optional arg for 'test_altdirs' to use to write the file number as it appears in its "
                         "source directory.")
parser.add_argument("--use_seen", type=bool, nargs="?", const=True, default=False,
                    help="Specify this if, given a file name, the script should seek out models to use which correspond "
                         "to the given arguments but specifically have seen the corresponding file name before; i.e. the "
                         "models are being assessed on subjects from data it has already seen.")
parser.add_argument("--use_balanced", type=str, nargs="?", const=True, default=False,
                    help="Specify this if, given a file name, the script should seek out models to use which correspond "
                         "to the given arguments but specifically are from a model trained on a 'class balanced' dataset.")
parser.add_argument("--single_act", type=int, nargs="?", const=True, default=False,
                    help="Specify this is intending to use the single-act files that are produced by 'mat_act_div' to "
                         "predict with. Only able to be used when 'dir'=NSAA.")
parser.add_argument("--single_act_concat", type=str, nargs="?", const=True, default=False,
                    help="Specify this if intending to use the single-act-concat files that are produced by "
                         "'mat_act_div' to predict with. Only able to be used when 'dir'=NSAA. If set to 'src_sac', "
                         "source files from the 'act_files_concat' directory, whereas if set to 'src_normal', source "
                         "files from there 'normal' location (i.e. as if the '--single_act_concat' arg wasn't set.")
parser.add_argument("--use_frc", type=bool, nargs="?", const=True, default=False,
                    help="Option to use the 'FRC_' files for 'AD' files instead of 'FR_' files, i.e. use files where "
                         "dimensionality reduction has been applied the same way over all files rather than on a "
                         "file-by-file basis.")
parser.add_argument("--standardize", type=bool, nargs="?", const=True, default=False,
                    help="Option to use models that have the '--standardize' argument set. Selects the relevant models "
                         "from the 'rnn_models' directory along with standardizing the file's data.")
parser.add_argument("--noise", type=str, nargs="?", const=True, default=False,
                    help="Option to use models that have the '--noise' argument set, which selects the relevant models "
                         "from the 'rnn_models' directory, but does not add noise to the data now being predicted.")
parser.add_argument("--batch", type=bool, nargs="?", const=True, default=False,
                    help="Option that is only set if the script is run from a batch file to access the external files "
                         "in a correct way.")
parser.add_argument("--use_ft_concat", type=str, nargs="?", const=True, default=False,
                    help="Specify this is the user wishes to load models that have had file types concatenated "
                         "together (i.e. where the 'ft' arg to 'rnn.py' was multiple file types separated by commas).")
parser.add_argument("--use_indiv", type=bool, nargs="?", const=True, default=False,
                    help="Specify this if wish to use models that have been trained only on the 'indiv' output type "
                         "when only using 'single_act' files to test w/ the model.")
parser.add_argument("--other_lo", type=str, nargs="?", const=True, default=False,
                    help="Specify this with a name of other files to have been left out of the model training and "
                         "testing data set. For example, with 'fn'=D3 and '--other_lo'=D10, search for models that "
                         "have been built containing '--leave_out=D3,D10' in the title. Separate multiple "
                         "files by commas.")
parser.add_argument("--add_dir", type=str, nargs="?", const=True, default=False,
                    help="Specify this with the name of another 'dir' if we wish to use models that are trained from "
                         "both 'dir' and '--add_dir' types. For example, if 'dir'=NSAA and '--add_dir_6MW-matFiles', "
                         "ensures that both 'NSAA' and '6MW-matFiles' directory files are used to train the models "
                         "with which we are concerned.")
parser.add_argument("--ft_red", type=str, nargs="?", const=True, default=False,
                    help="Specify this is using models that have had their raw measurement inputs feature reduced; note "
                         "that this, like '--use_ft_concat', loads '--pca' models but specifically not models with "
                         "their input features concatenated.")
parser.add_argument("--single_sequence", type=bool, nargs="?", const=True, default=False,
                    help="Specify this if we only wish to draw a single sequence from the source file(s) for the subject "
                         "that is drawn evenly across the complete file(s). Note that this is replicated 'batch_size' "
                         "times so it can be accepted into the pre-built models.")
parser.add_argument("--combine_preds", type=bool, nargs="?", const=True, default=False,
                    help="Specify this if we wish to combine the predictions made using models built for different "
                         "output types to get an aggregate prediction for the overall NSAA score.")
parser.add_argument("--no_testset", type=bool, nargs="?", const=True, default=False,
                    help="Specify this to select models that have the train/test ratio set to 0.")
parser.add_argument("--new_subject", type=bool, nargs="?", const=True, default=False,
                    help="Specify this if the we wish to treat the subject, 'fn', as one with no 'true' y-labels. "
                         "Hence, specify this if the subject does not have 'true' values recorded by specialists that "
                         "is stored in the 'nsaa_6mw_info.xlsx' file (e.g. if the subject is a new subject with no "
                         "true labels or if we otherwise wish to treat them as such).")
parser.add_argument("--leave_out_version", type=str, nargs="?", const=True, default=False,
                    help="Specify this with the name of a version of subjects (e.g. 'V2') to assess the 'fn' file "
                         "on models with the specified version of files not used to train the models.")
args = parser.parse_args()


#Ensures 'dir'=NSAA when '--single_acts' is set
if args.single_act and args.dir != "NSAA":
    print("First arg ('dir') must be 'NSAA' when '--single_acts' is set, as there are only single acts files from "
          "the files from the 'NSAA' directory.")
    sys.exit()

#Ensures 'dir'=NSAA when '--single_acts_concat' is set
if args.single_act_concat and args.dir != "NSAA":
    print("First arg ('dir') must be 'NSAA' when '--single_acts_concat' is set, as there are only single acts files from "
          "the files from the 'NSAA' directory.")
    sys.exit()


#Appends the sub_dir name to 'source_dir' if it's one of the allowed names
if args.dir + "\\" in sub_dirs:
    source_dir += args.dir.upper() + "\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'direct_csv', 'allmatfiles', or 'left-out'.")
    sys.exit()

#Ensures the corresponding second argument when dealing with files from 'allmatfiles'
if args.dir == "allmatfiles" and args.ft != "jointAngle":
    print("Second arg ('ft') must be 'jointAngle when 'dir' argument is \'allmatfiles\'")
    sys.exit()

#Ensures that, if '--balance' is set, it is either 'up' to upsample the data or 'down' to downsample the data
if args.use_balanced:
    if not args.use_balanced == "up" and not args.use_balanced == "down":
        print("Optional arg ('--balance') must be set to either 'up' or 'down'.")
        sys.exit()

#Ensures that, if '--single_act_concat' is set, it is one of the two allowed strings
if args.single_act_concat:
    if not args.single_act_concat == "src_sac" and not args.single_act_concat == "src_normal":
        print("'--single_act_concat' must be set to one of 'use_sac' or 'use_normal'...")
        sys.exit()

fts, sds = [], []
for ft in args.ft.split(","):
    fts.append(ft)
    if args.dir == "left-out" and ft == "AD":
        sds.append(local_dir + "output_files\\left-out\\AD\\")
    elif args.dir == "left-out" and ft in file_types:
        sds.append(local_dir + "left-out\\" + ft + "\\")
    elif ft + "\\" in sub_sub_dirs and not args.single_act and not args.single_act_concat:
        sds.append(source_dir + ft + "\\")
    elif ft + "\\" in sub_sub_dirs and args.single_act:
        sds.append(source_dir + ft + "\\act_files\\")
    elif ft + "\\" in sub_sub_dirs and args.single_act_concat=="src_sac":
        sds.append(source_dir + ft + "\\act_files_concat\\")
    elif ft + "\\" in sub_sub_dirs and args.single_act_concat == "src_normal":
        sds.append(source_dir + ft + "\\")
    elif ft in file_types and args.dir == "NSAA" and args.single_act_concat=="src_sac":
        sds.append(local_dir + "NSAA\\matfiles\\act_files_concat\\" + ft + "\\")
    elif ft in file_types and args.dir == "NSAA" and args.single_act_concat == "src_normal":
        sds.append(local_dir + "NSAA\\matfiles\\" + ft + "\\")
    elif ft in file_types and args.dir == "NSAA" and not args.single_act:
        sds.append(local_dir + "NSAA\\matfiles\\" + ft + "\\")
    elif ft in file_types and args.dir == "NSAA" and args.single_act:
        sds.append(local_dir + "NSAA\\matfiles\\act_files\\" + ft + "\\")
    elif ft in file_types and args.dir == "NSAA" and not args.single_act_concat:
        sds.append(local_dir + "NSAA\\matfiles\\" + ft + "\\")
    elif args.dir == "allmatfiles":
        sds.append(local_dir + "allmatfiles\\" + ft + "\\")
    elif args.dir == "NMB" and ft != "AD":
        sds.append(local_dir + "NMB\\" + ft + "\\")
    elif args.dir == "NMB":
        sds.append(source_dir + "NMB\\AD\\")
    else:
        print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
              "\'JA', or \'DC\' (unless dir is give as 'NSAA', where 'ft' can be a measurement name), and each part "
              "must be separated by a comma for each measurement to use in the ensemble.")
        sys.exit()

#Adds a dash to the beginning of the 'fn' argument input if the '--handle_dash' optional argument is set
if args.handle_dash:
    args.fn = "-" + args.fn

local_nsaa_6mw_path = "..\\" + nsaa_6mw_path if args.batch else nsaa_6mw_path

#Section below encompasses all the necessary rules to fetch files from various directories, subdirectories, and so on
#based on the 'dir', 'fn', 'ft', and other option arguments. Result of this section is a list of short file names, 'fns',
#that contain the list of all names of files (not including their full paths)
fns = []
for ft, sd in zip(fts, sds):
    if args.dir == "allmatfiles":
        if any(args.fn.upper() == s.split("jointangle")[-1].split(".mat")[0].upper() for s in os.listdir(sd)):
            fns.append([s for s in os.listdir(sd) if args.fn.upper() == s.split("jointangle")[-1].split(".mat")[0].upper()][0])
        else:
            print("Cannot find '" + str(args.fn) + "' in '" + sd + "'")
            sys.exit()
    elif args.dir == "left-out":
        fns.append([s for s in os.listdir(sd) if args.fn.split("-")[0] in s][0])
    elif args.dir == "NMB" and ft != "AD":
        fns.append([s for s in os.listdir(sd) if args.fn in s][0])
    elif args.dir == "NMB":
        fns.append([s for s in os.listdir(sd) if "FR_AD_" + args.fn.split("-")[0] + "_" in s][0])
    elif ("AD\\" in sd and any(args.fn.upper().split("-")[0] == s.upper().split("_")[2] for s in os.listdir(sd)
                               if s.endswith(".csv"))) \
            or any(args.fn.upper() == "-".join(s.upper().split("-")[:2]) for s in os.listdir(sd)) \
            or any(args.fn.upper() == s.upper().split("-")[0] for s in os.listdir(sd)) \
            or any(args.fn.upper() == s.upper().split("_")[2] for s in os.listdir(sd) if s.endswith(".csv")):
        if "AD\\" in sd:
            if not args.single_act:
                fns.append([s for s in os.listdir(sd) if args.fn.upper().split("-")[0] == s.upper().split("_")[1]][0])
            else:
                fns.append([s for s in os.listdir(sd) if args.fn.upper() == s.upper().split("_")[1]
                            and "act" + str(args.single_act) + "_" in s][0])
        else:
            if not args.single_act:
                fns.append([s for s in os.listdir(sd) if args.fn in s][0])
            else:
                try:
                    fns.append([s for s in os.listdir(sd) if args.fn in s and "act" + str(args.single_act) + "_" in s][0])
                except IndexError:
                    print("Act " + str(args.single_act) + " not found for " + args.fn + " in '" + sd + "', skipping...")
                    sys.exit()
    elif ft == "AD" and args.single_act:
        fns.append([s for s in os.listdir(sd) if args.fn.upper() == s.upper().split("_")[1]
                    and "act" + str(args.single_act) + "_" in s][0])
    elif args.single_act_concat:
        try:
            fns.append([s for s in os.listdir(sd) if args.fn.upper() == s.upper().split("_")[0]][0])
        except IndexError:
            print("Cannot find subject '" + args.fn.upper() + "' in " + sd + "...")
            sys.exit()
    else:
        print("Cannot find '" + str(args.fn) + "' in '" + sd + "'")
        sys.exit()


#Creates a list of the full file path names to load the models of, given the type of file and short file names ('fns')
full_file_names = []
for sd, fn in zip(sds, fns):
    if fn.startswith("AD"):
        if not args.use_frc:
            full_file_names.append(sd + "FR_" + fn)
        else:
            full_file_names.append(sd + "FRC_" + fn)
    elif args.dir == "allmatfiles":
        full_file_names.append(sd + "jointAngle\\" + fn.split(".")[0] + "_jointAngle.csv")
    else:
        full_file_names.append(sd + fn)


#Specifies the directories (i.e. names of file groupings that can be used to train a model, e.g. 'NSAA' or
#'6MW-matFiles') on which to test the file. If the 'alt_dirs' optional argument is not provided, then 'dir' is used
#(i.e. the file in question is tested on the same directory type as it comes from).
if args.alt_dirs:
    search_dirs = args.alt_dirs.split("_")
    if not all(sd + "\\" in sub_dirs for sd in search_dirs):
        print("Optional arg ('--alt_dirs') must be directory names separated by commas and each must be one of "
              "'NSAA', '6minwalk-matfiles', '6MW-matFiles', or 'direct_csv'.")
        sys.exit()
else:
    search_dirs = [args.dir]

#The models that are to be tested by the script are selected based on their names; e.g. if 'dir'=NSAA and 'ft'=position,
#then 'NSAA_position_all_acts', 'NSAA_position_all_dhc', and 'NSAA_position_all_overall' will be loaded as directory
#names containing the trained models
models = []
for sd in search_dirs:
    inner_models = []
    if args.use_indiv:
        output_types = ["indiv"]
    for ot in output_types:
        inner_inner_models = []
        #If the '--use_ft_concat' arg is set, for each of the output types, find the model that contains all of the
        #file types that are given to 'model_predictor' within its title
        if args.use_ft_concat:
            match_dirs = [fn for fn in os.listdir(model_dir) if fn.split("_")[0] == sd and
                          fn.split("_")[3] == ot]
            for md in match_dirs:
                if "--pca=" + args.use_ft_concat in md and all(ft for ft in args.ft.split(",") if ft in md):
                    if not args.use_seen:
                        if ("--leave_out=" + args.fn) in md:
                            inner_inner_models.append(md)
                            print("Using model: '" + str(md) + "'...")
                            break
                    else:
                        if "--leave_out=" not in md:
                            inner_inner_models.append(md)
                            print("Using model: '" + str(md) + "'...")
                            break
            inner_models.append(inner_inner_models)
        elif args.ft_red:
            match_dirs = [fn for fn in os.listdir(model_dir) if fn.split("_")[0] == sd and
                          fn.split("_")[3] == ot and "--pca=" + args.ft_red in fn]
            for ft in fts:
                for md in match_dirs:
                    if not args.use_seen:
                        if ("--leave_out=" + args.fn) in md and md.split("_")[1] == ft:
                            inner_inner_models.append(md)
                            print("Using model: '" + str(md) + "'...")
                    else:
                        if "--leave_out=" not in md and md.split("_")[1] == ft:
                            inner_inner_models.append(md)
                            print("Using model: '" + str(md) + "'...")
            inner_models.append(inner_inner_models)
        else:
            for i, ft in enumerate(fts):
                #Handles the special case when we're using NSAA files that are placed in the 'left-out' source dir
                if args.dir == "left-out":
                    inner_inner_models.append([md for md in os.listdir(model_dir) if md.split("_")[0] == "NSAA"
                                               and md.split("_")[1] == ft and md.split("_")[3] == ot
                                               and "--leave_out" not in md and "--balance" not in md
                                               and "--other_dir" not in md][0])
                elif any(fn for fn in os.listdir(model_dir) if fn.split("_")[0] == sd and
                                                             fn.split("_")[3] == ot and fn.split("_")[1] == ft):
                    #Gets the first inner model directory that match the file type, directory name, and output type, with
                    #additional preference of model trained on 'left out file' if one exists in 'model_dir. Also
                    #ensures that if '--add_dir' is used that its value is within the directory name, otherwise ensures
                    #that any models containing '--add_dir' values aren't used.
                    if not args.add_dir:
                        match_dirs = [fn for fn in os.listdir(model_dir) if fn.split("_")[0] == sd and
                                      fn.split("_")[3] == ot and fn.split("_")[1] == ft]
                    else:
                        match_dirs = [fn for fn in os.listdir(model_dir) if fn.split("_")[3] == ot and
                                      fn.split("_")[1] == ft and sd in fn.split("_")[0] and
                                      args.add_dir in fn.split("_")[0]]
                    #Ensures that if '--other_lo' is used that its value is within the directory name, otherwise ensures
                    #that any models containing 'args.fn + ','' (i.e. models with more than 1 file left out) aren't used
                    match_dirs = [fn for fn in match_dirs if args.other_lo in fn] if args.other_lo \
                        else [fn for fn in match_dirs if args.fn + "," not in fn]
                    match_dirs = [fn for fn in match_dirs if "--use_sac" in fn] if args.single_act_concat \
                        else [fn for fn in match_dirs if "--use_sac" not in fn]
                    match_dirs = [fn for fn in match_dirs if "--no_testset" in fn] if args.no_testset \
                        else [fn for fn in match_dirs if "--no_testset" not in fn]
                    match_dirs = [fn for fn in match_dirs if "--leave_out_version=" + args.leave_out_version in fn] \
                        if args.leave_out_version \
                        else [fn for fn in match_dirs if "--leave_out_version=" not in fn]
                    if any(args.fn in md for md in match_dirs):
                        #If 'use_balanced' is set, specifically use ones which have '--balance' in directory name
                        if args.use_balanced and args.use_balanced == "up":
                            if not args.use_seen and not args.leave_out_version:
                                #If 'use_seen' is not set, specifically use ones that have '--leave_out='fn'' in directory name
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--balance=up" in md and "--leave_out=" + args.fn in md][0])
                            else:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--balance=up" in md and "--leave_out=" not in md][0])
                        elif args.use_balanced and args.use_balanced == "down":
                            if not args.use_seen and not args.leave_out_version:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--balance=down" in md and "--leave_out=" + args.fn in md][0])
                            else:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--balance=down" in md and "--leave_out=" not in md][0])
                        elif args.use_frc and ft == "AD":
                            if not args.use_seen and not args.leave_out_version:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--use_frc" in md and "--leave_out=" + args.fn in md][0])
                            else:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--use_frc" in md and "--leave_out=" not in md][0])
                        elif args.standardize:
                            if not args.use_seen and not args.leave_out_version:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--standardize" in md and "--leave_out=" + args.fn in md][0])
                            else:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--standardize" in md and "--leave_out=" not in md][0])
                        elif args.noise:
                            if not args.use_seen and not args.leave_out_version:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--noise=" + args.noise in md and "--leave_out=" + args.fn in md][0])
                            else:
                                inner_inner_models.append([md for md in match_dirs
                                                           if "--noise=" + args.noise in md and "--leave_out=" not in md][0])
                        else:
                            option_args = ["--balance=up", "--balance=down", "--use_frc", "--use_frc",
                                           "--standardize", "--noise"]
                            filtered_match_dirs = [md for md in match_dirs
                                                   if not any(opt_arg in md for opt_arg in option_args)]
                            if not args.use_seen and not args.leave_out_version:
                                inner_inner_models.append([md for md in filtered_match_dirs if "--leave_out=" + args.fn in md][0])
                            else:
                                inner_inner_models.append([md for md in filtered_match_dirs if "--leave_out=" not in md][0])
                    else:
                        inner_inner_models.append(match_dirs[0])
                else:
                    inner_inner_models.append(None)
                print("Using model: '" + str(inner_inner_models[-1]) + "'...")
            inner_models.append(inner_inner_models)
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
model_shapes_path = "..\\" + model_shapes_path if args.batch else model_shapes_path
model_shape = pd.read_excel(model_shapes_path)

if args.add_dir:
    sequence_lengths = [[[model_shape.loc[(model_shape["dir"] == model.split("_")[0].split(",")[0]) &
                                          (model_shape["ft"] == model.split("_")[1]) &
                                          (model_shape["measure"] == model.split("_")[3])].iloc[
                              -1, -1] if model else None
                          for model in inner_inner_models] for inner_inner_models in inner_models] for inner_models in
                        models]
elif not args.use_ft_concat:
    sequence_lengths = [[[model_shape.loc[(model_shape["dir"] == model.split("_")[0]) &
                                          (model_shape["ft"] == model.split("_")[1]) &
                                         (model_shape["measure"] == model.split("_")[3])].iloc[-1, -1] if model else None
                         for model in inner_inner_models] for inner_inner_models in inner_models] for inner_models in models]
else:
    sequence_lengths = [[[model_shape.loc[(model_shape["dir"] == model.split("_")[0]) &
                                          (model_shape["measure"] == model.split("_")[3])].iloc[-1, -1] if model else None
                          for model in inner_inner_models] for inner_inner_models in inner_models] for inner_models in
                        models]


def preprocessing(full_file_name, sequence_length):
    #Loads the .csv data from the given file name and divides it up into a 3D format based on 'sequence_length'; for
    #example, if 'df_x' is of shape (1000, 50) and 'sequence_length' = 60, then 'x_data' becomes (16, 60, 50). Note that
    #the 'leftover' data at the end of 'df_x' that does not fit into a (60, 50) shape is discarded.
    if not "AD" in full_file_name.split("\\")[-1]:
        df_x = pd.read_csv(full_file_name, index_col=0).values
    else:
        df_x = pd.read_csv(full_file_name, index_col=0).iloc[:, 20:].values

    #If we only wish to use a single sequence drawn from the subject file, draw 'sequence_length' frames of data
    #from the file that is evenly distributed across the file
    if args.single_sequence:
        df_x = np.array([df_x[i] for i in range(0, len(df_x), int(len(df_x)/sequence_length)+1)])


    x_data = [df_x[sequence_length*i:(sequence_length*(i+1)), :] for i in range(int(len(df_x)/sequence_length))]

    #Handles the case where the '.mat' file doesn't even contain enough data for one sequence (i.e. the number of rows
    #in the file is < 'sequence_length')
    if len(x_data) == 0:
        print(full_file_name.split("\\")[-1].split(".")[0], "too short in length to use, skipping....")
        sys.exit()

    #If the '--new_subject' optional arg is split, returns without fetching the 'y' label information
    #from the 'nsaa_6mw_info.xlsx' file
    if args.new_subject:
        return x_data, None, None, None

    #Loads the file that contains information on each subject and their corresponding overall and individual NSAA scores,
    #and three 'true' labels for the file are extracted: 'y_label_dhc' gets 1 if the short name of the file (e.g. 'D11')
    #begins with a 'D', else it gets a 0; 'y_label_overall' gets an integer value between 0 and 34 of the overall NSAA score,
    #and 'y_label_acts' gets the 17 individiual NSAA acts as a list as scores between 0 and 2
    df_y = pd.read_excel(local_nsaa_6mw_path)
    if args.dir != "direct_csv" and not source_dir.startswith(local_dir):
        y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[2][0].upper() == "D" else 0
    elif source_dir.startswith(local_dir):
        if "FR_" in full_file_name or "FRC_" in full_file_name:
            y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[2][0].upper() == "D" else 0
        elif full_file_name.split("\\")[-1].startswith("jointangle"):
            y_label_dhc = 1 if full_file_name.split("\\")[-1].split("angle")[1][0].upper() == "D" else 0
        else:
            y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[0][0].upper() == "D" else 0
    else:
        y_label_dhc = 1 if full_file_name.split("\\")[-1].split("_")[1][0].upper() == "D" else 0
    y_index = df_y.index[df_y["ID"] == args.fn.upper().split("-")[0]]
    #Raises an exception
    if len(y_index) == 0:
        #raise FileExistsError
        print(args.fn.upper().split("-")[0] + " not an entry in 'nsaa_6mw_info.xlsx' for " + full_file_name + ", skipping...")
        sys.exit()
    y_label_overall = df_y.loc[y_index, "NSAA"].values[0]
    try:
        y_label_acts = [int(num) for num in df_y.iloc[y_index, 5:].values[0]]
    except ValueError:
        print("\nEntry for '" + args.fn + "' single-act scores not in table, skipping file...\n")
        sys.exit()


    #If the '--standardize' optional argument is given, reshape the 'x' data into a 2D array, standardize
    #each of the features, and reshape it back into its original shape (note: np.concatenate and np.reshape aren't used
    #in order to ensure the data is written back as we would expect with the same number of lines)
    if args.standardize:
        x_data = [sample for sequence in x_data for sample in sequence]
        print("Standardizing the data set...")
        x_data = StandardScaler().fit_transform(x_data)
        x_data = [[x_data[i * x_shape[1] + j] for j in range(x_shape[1])] for i in range(x_shape[0])]

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



def combine_preds(output_strs):
    """
        :return: gets the results that are contained within the output strings, computes a method for predicting an
        overall NSAA score based on the D/HC label and percentage of predicted sequences, and aggregates the
        predictions made by all three output types to get an average aggregated overall NSAA score prediction
    """

    #Gets the numerical predictions made for each of the output metrics as contained within 'output_strs'
    acts_sum_pred = sum(eval([out_str for out_str in output_strs if "Predicted 'Acts Sequence'" in out_str][0].split(" = ")[1]))
    dhc_pred = [out_str for out_str in output_strs if "Predicted 'D/HC Label'" in out_str][0].split(" = ")[1]
    dhc_prop_pred = float([out_str for out_str in output_strs if "Percentage of predicted '" + dhc_pred + "' sequences"
                in out_str][0].split(" = ")[1].split("%")[0])/100
    overall_pred = int([out_str for out_str in output_strs if "Predicted 'Overall NSAA Score'" in out_str][0].split(" = ")[1])

    #Gets the median value of all the subjects true overall NSAA scores according to the 'nsaa_6mw_info.xlsx' file
    #and uses this as the basis for the overall NSAA score associated with the 'D' label
    df_y = pd.read_excel(local_nsaa_6mw_path)
    df_y = df_y.loc[df_y["ID"].str.startswith("D")]
    d_median = int(np.median(df_y["NSAA"].tolist()))

    #Averages the overall NSAA scores made by each of the predictions, with the average of the predicted overall NSAA
    #score, the sum of the indivudal acts predictions, and the assigned numerical value for the 'D' or 'HC' label
    dhc_value = 34 if dhc_pred == "HC" else d_median
    average_pred = round((acts_sum_pred + overall_pred + (dhc_value*dhc_prop_pred)) / (2 + dhc_prop_pred))

    output_strs.append("Aggregated predicted 'Overall NSAA Score' = " + str(average_pred))

    return output_strs



pred_overalls, true_overalls = [], []
#Stores the strings to write AFTER all the models have run so all results are printed at the end, rather than some being
#printed and then obscured by the model setup text
output_strs = []
#For each directory type that we wish to assess on...
for i, inner_models in enumerate(models):
    output_strs.append("\t\t ------ " + str(search_dirs[i]) + " Predictions ------ ")
    #For each output type that we are training towards (e.g. 'overall', 'acts', etc.)...
    for j, inner_inner_models in enumerate(inner_models):
        #'output_type' contains one of 'acts', 'indiv', or 'dhc'`
        output_type = None
        for model in inner_inner_models:
            if model:
                output_type = model.split("_")[3]
        preds = []
        y_label_dhc, y_label_overall, y_label_acts = (None,)*3
        #For each measurement type (i.e. for every model name, e.g. 'jointAngle' or 'AD')
        ft_concat_data = [[] for i in range(5)]
        data_entered = False
        for k, full_file_name in enumerate(full_file_names):
            #If the argument is set then, for each of the file types given, concatenate the data together
            #and if it's the last file type in the sequence of given file types, reduce the dimensions of the sequences
            #to the value given by the argument
            if args.use_ft_concat:
                x_data, y_label_dhc, y_label_overall, y_label_acts = preprocessing(full_file_name, sequence_lengths[0][j][0])
                ft_concat_data[0] = np.concatenate((ft_concat_data[0], x_data), axis=2) if data_entered else x_data
                ft_concat_data[1] = ft_concat_data[1] if data_entered else y_label_dhc
                ft_concat_data[2] = ft_concat_data[2] if data_entered else y_label_overall
                ft_concat_data[3] = ft_concat_data[3] if data_entered else y_label_acts
                data_entered = True
                if k == len(full_file_names)-1:
                    x_data, y_label_dhc, y_label_overall, y_label_acts = \
                        ft_concat_data[0], ft_concat_data[1], ft_concat_data[2], ft_concat_data[3]
                    try:
                        pca = int(args.use_ft_concat)
                        x_shape = np.shape(x_data)
                        x_data = [sample for sequence in x_data for sample in sequence]
                        x_data, y_data = ft_red_select(x_data, None, "pca", False, False, pca)
                        x_data = [[x_data[i * x_shape[1] + j] for j in range(x_shape[1])] for i in range(x_shape[0])]
                    except ValueError:
                        if args.use_ft_concat != "all":
                            print("Optional arg '--pca' must contain int value or 'all' to keep all features...")
                            sys.exit()
                else:
                    continue
            #Else, prepare the data as normal for the given output type and file
            else:
                #If there is a 'None' sequence length (i.e. a corresponding model for the output type and file type doesn't
                #exist), then skip over it
                if not sequence_lengths[i][j][k]:
                    continue
                x_data, y_label_dhc, y_label_overall, y_label_acts = preprocessing(full_file_name, sequence_lengths[i][j][k])
                if args.ft_red:
                    try:
                        pca = int(args.ft_red)
                        x_shape = np.shape(x_data)
                        x_data = [sample for sequence in x_data for sample in sequence]
                        x_data, y_data = ft_red_select(x_data, None, "pca", False, False, pca)
                        x_data = [[x_data[i * x_shape[1] + j] for j in range(x_shape[1])] for i in range(x_shape[0])]
                    except ValueError:
                        if args.use_ft_concat != "all":
                            print("Optional arg '--pca' must contain int value or 'all' to keep all features...")
                            sys.exit()



            #Complete model path to the directory of the model in question
            model_path = model_dir + inner_inner_models[k] if not args.use_ft_concat else model_dir + inner_inner_models[0]
            inner_preds = []
            with tf.Session(graph=tf.Graph()) as sess:
                #Loads the model and restores from it's final checkpoint
                new_saver = tf.train.import_meta_graph(model_path + "\\model.ckpt.meta", clear_devices=True)
                new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
                #If there aren't enough sequences within the file being tested to make up a full batch (i.e. len(x_data)
                #< batch_size), then replicate it until it's the size of at least batch size
                new_x_data = []
                while len(new_x_data) < batch_size:
                    new_x_data += list(x_data)
                x_data = new_x_data
                #For each batch of the data from 'full_file_name', feed it through the trained model to get predictions based
                #on the type of model (e.g. '1's or '0's if output_type == "dhc", ints between 0 and 34 if output_type ==
                #"overall", and lists of 17 values between 0 and 1 if output_type == "dhc") and appends these to 'preds'
                for ii, batch_x in enumerate(create_batch_generator(x_data, None, batch_size=batch_size), 1):
                    feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0}
                    if output_type == "overall" or args.use_indiv:
                        inner_preds.append(sess.run('logits_squeezed:0', feed_dict=feed))
                    else:
                        inner_preds.append(sess.run('labels:0', feed_dict=feed))
            #Flatten these values to a 1D list
            inner_preds = np.concatenate(inner_preds)
            #Gets rid of errant 'nan's in the predictions (cause unknown)
            inner_preds = [ip for ip in inner_preds if "nan" not in str(ip)]
            #Add these predictions to aggregate list of predictions for the output type in question
            preds.append(inner_preds)
            #Appends all predicted overall NSAA scores for a given model to the list containing the aggregated predicted scores
            if output_type == "overall":
                pred_overalls += inner_preds


        #Aggregate results for measurements
        if output_type == "acts":
            #Makes sure that each measurement's worth of predictions are the same length (e.g. if there are far
            #fewer 'AD' predictions than 'jointAngle', then duplicate 'AD' predictions to be of the same length)
            preds_lens = [len(preds[m]) for m in range(len(preds))]
            for m in range(len(preds)):
                #If the length of one part of 'preds' is shorter than the rest, add 'batch_size' chunks of that part
                #of preds until it matches that of the largest part. This ensures they all have the same length
                #to be able to use 'np.transpose()'.
                while len(preds[m]) < max(preds_lens):
                    preds[m] += preds[m][:batch_size]
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
            true_overalls = [y_label_overall for m in range(len(pred_overalls))]
            if not args.new_subject:
                output_strs.append(str("True 'Overall NSAA Score' = " + str(y_label_overall)))
            output_strs.append(str("Predicted 'Overall NSAA Score' = " + str(int(round(float(np.mean(preds)), 0)))))
        elif output_type == "dhc":
            true_label = "D" if y_label_dhc == 1 else "HC"
            pred_label = "D" if max(set(list(preds)), key=list(preds).count) == 1 else "HC"
            d_percen = np.round(((np.sum(preds)/len(preds))*100), 2)
            hc_percen = np.round(100 - d_percen, 2)
            if not args.new_subject:
                output_strs.append(str("True 'D/HC Label' = " + true_label))
            output_strs.append(str("Predicted 'D/HC Label' = " + pred_label))
            output_strs.append(str("Percentage of predicted 'D' sequences = " + str(d_percen) + "%"))
            output_strs.append(str("Percentage of predicted 'HC' sequences = " + str(hc_percen) + "%"))
        elif args.use_indiv:
            if not args.new_subject:
                true_label = y_label_acts[int(args.single_act)-1]
                output_strs.append(str("True individual activity score = " + str(true_label)))
            output_strs.append(str("Predicted individual activity score = " + str(round(float(np.mean(preds)), 2))))
        else:
            preds = np.transpose(preds)
            pred_acts = [Counter(preds[i]).most_common()[0][0] for i in range(len(preds))]
            perc_correct = 0
            if not args.new_subject:
                num_correct = 0
                for m in range(len(y_label_acts)):
                    if y_label_acts[m] == pred_acts[m]:
                        num_correct += 1
                perc_correct = round(((num_correct / 17)*100), 2)
                output_strs.append(str("True 'Acts Sequence' = " + str(y_label_acts)))
            output_strs.append(str("Predicted 'Acts Sequence' = " + str(pred_acts)))
            if not args.new_subject:
                output_strs.append(str("Percent of acts correctly predicted = " + str(perc_correct) + "%"))
        output_strs.append("")


#If the optional '--combine_preds' argument is set, compute the aggregated predicted 'Overall NSAA Score' and add this
#to the list of output strings
if args.combine_preds:
    output_strs = combine_preds(output_strs)


#Prints to the user all the output lines that were generated by testing on all the models
print("\n")
for output_str in output_strs:
    print(output_str)



#Plot the predicted values against the true values for NSAA overall (only if *not* run from 'test_altdirs.py'; does not
#worth if '--new_subject' is set, as won't have the 'true' values
if not args.file_num and not args.new_subject:
    fig, ax = plt.subplots()
    ax.scatter(true_overalls, pred_overalls, alpha=0.03)
    plt.xlim(0, 34)
    plt.ylim(0, 34)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.title("Plot of true overall NSAA scores against predicted overall NSAA scores")
    plt.xlabel("True overall NSAA scores")
    plt.ylabel("Predicted overall NSAA scores")
    graphs_path = "..\\..\\documentation\\Graphs\\Model_predictor_" + args.dir + "_" + args.ft + "_" + args.fn if args.batch \
        else "..\\documentation\\Graphs\\Model_predictor_" + args.dir + "_" + args.ft + "_" + args.fn
    plt.savefig(graphs_path)
    plt.gcf().set_size_inches(10, 10)
    if args.show_graph:
        plt.show()

#Copies the 'fn' argument value to 'write_fn', which will be modified and then added to the list of 'output_strs'
write_fn = args.fn
#Appends a short string to the file name being written to file if the models have already seen the file name in training
if args.use_seen:
    write_fn += " (already seen)"
if args.other_lo:
    write_fn += " (other lo = " + args.other_lo + ")"
if args.single_act_concat == "src_sac":
    write_fn += " (src sac)"
elif args.single_act_concat == "src_normal":
    write_fn += " (src normal)"
if args.use_ft_concat:
    write_fn += " (feature concat = " + args.use_ft_concat + ")"
if args.ft_red:
    write_fn += " (feature reduced = " + args.ft_red + ")"
if args.add_dir:
    write_fn += " (additional dir = " + args.add_dir + ")"
if args.single_act:
    write_fn += " (act " + str(args.single_act) + ")"
if args.single_sequence:
    write_fn += " (single sequence)"
if args.combine_preds:
    write_fn += " (aggregate overall)"
if args.no_testset:
    write_fn += " (no testset)"
if  args.use_indiv:
    write_fn += " (indiv)"
if args.use_frc:
    write_fn += " (FRC)"
if args.noise:
    write_fn += " (noise = " + args.noise + ")"
if args.use_balanced and args.use_balanced == "down":
    write_fn += " (downsampled)"
elif args.use_balanced and args.use_balanced == "up":
    write_fn += " (upsampled)"
if args.leave_out_version:
    write_fn += " (leave out version = " + args.leave_out_version + ")"


#If we're predicting on a subject where we know the true values of their overall score, individual scores, etc., then
#we write to 'model_predictions.csv' as this is a table that stores the true values for each output type as well as predicted
if not args.new_subject:
    #Creates a list of outputs of the model in a way more fitting of a .csv file (e.g. reducing writing "Percent correct = 10%"
    #to "10%" when writing it to a cell, while the corresponding header becomes "Percent correct")
    header, new_output_strs = [], []
    for out_str in output_strs:
        if out_str == "":
            pass
        elif " = " in out_str:
            header.append(out_str.split(" = ")[0])
            new_output_strs.append(out_str.split(" = ")[1])
        else:
            header.append(out_str)
            new_output_strs.append(out_str)

    #Adds in additional strings to the row that are based on the arguments based in to 'model_predictor' and also creates
    #additional corresponding header strings
    header = ["Short file name", "Source dir", "Model trained dir(s)", "Measurements tested"] + header
    if args.alt_dirs:
        new_output_strs = [write_fn, args.dir, args.alt_dirs.split("_"), args.ft.split(",")] + new_output_strs
    else:
        #If same directory is used for training models as is used for assessing, add a 'N/A' to the column
        new_output_strs = [write_fn, args.dir, "N/A", args.ft.split(",")] + new_output_strs

    #Adds a index name that is set to what number it corresponds to within 'dir' if passed from 'test_altdirs'
    #(e.g. '8/470') or 'N/A' if the optional '--file_num' arg is not provided
    ind_lab = args.file_num if args.file_num else "N/A"
    #Creates a single-line DataFrame from the predictions made by the model(s) with corresponding headers
    output_strs_df = pd.DataFrame([new_output_strs], columns=header, index=[ind_lab])


    #Writes the single-line DataFrame to a new .csv file if it doesn't exist or, if it does exist, appends it to the end
    #of the existing one
    model_pred_path = "..\\" + model_pred_path if args.batch else model_pred_path
    if not os.path.exists(model_pred_path):
        with open(model_pred_path, 'w', newline='') as file:
            output_strs_df.to_csv(file, header=header)
    else:
        with open(model_pred_path, 'a', newline='') as file:
            output_strs_df.to_csv(file, header=False)

#If '--new_subject' is set (i.e. we're using a subject with a new version or a completely new subject altogether that
#doesn't exist within the 'nsaa_6mw_info' table), then write to 'model_predictions_newfiles.csv'
else:
    df = pd.read_excel(local_nsaa_6mw_path)

    #Gets the ID of the row we are fetching in the table. If 'fn' is a different version of a subject that exists in
    #the table (e.g. 'fn'=D4V2), then we retrieve the subject's previous version's info (e.g. the 'D4' row).
    id = args.fn.split("-")[0][:-2] if "V2" in args.fn else args.fn.split("-")[0]
    df_row = df.loc[df["ID"] == id]

    #Gets the overall score, individual scores, and D/HC classification of the 'comparison' entry in the table if one
    #exists (e.g. if the 'fn' args is a 'V2' file with a 'V1' file entry in 'nsaa_6mw_info.xlsx'), else if not in the
    #table then add corresponding '(Not in table)' values
    dhc_label = ("D" if id.startswith("D") else "HC") if not df_row.empty else "(Not in table)"
    overall_score = int(df_row.iloc[0, 4]) if not df_row.empty else "(Not in table)"
    acts_scores = df_row.iloc[0, 5:22].astype(float).values.tolist() if not df_row.empty else "(Not in table)"
    acts_scores = [int(act_score) for act_score in acts_scores]


    #Gets the values that we have predicted for the 'fn' file we are predicting upon
    dhc_pred = [out_str for out_str in output_strs if "Predicted 'D/HC Label'" in out_str][0].split(" = ")[1]
    pred_str = "Aggregated predicted 'Overall NSAA Score'" if args.combine_preds else "Predicted 'Overall NSAA Score'"
    overall_pred = int([out_str for out_str in output_strs if pred_str in out_str][0].split(" = ")[1])
    acts_pred = [out_str for out_str in output_strs if "Predicted 'Acts Sequence'" in out_str][0].split(" = ")[1]

    header = ["Short file name", "Source dir", "Model trained dir(s)", "Measurements tested",
              "True D/HC label (other version)", "Predicted D/HC label (short file name)",
              "True overall NSAA score (other version)", "Predicted overall NSAA Score (short file name)",
              "True single act scores (other version)", "Predicted single acts scores (short file name)"]

    new_output_strs = [write_fn, args.dir, args.ft.split(","), dhc_label, dhc_pred, overall_score,
                       overall_pred, acts_scores, acts_pred]

    if args.alt_dirs:
        new_output_strs.insert(2, args.alt_dirs.split("_"))
    else:
        new_output_strs.insert(2, "N/A")

    output_strs_df = pd.DataFrame([new_output_strs], columns=header)

    model_pred_newfiles_path = model_pred_path.split(".csv")[0] + "_newfiles.csv"
    model_pred_newfiles_path = "..\\" + model_pred_newfiles_path if args.batch else model_pred_newfiles_path

    if not os.path.exists(model_pred_newfiles_path):
        with open(model_pred_newfiles_path, 'w', newline='') as file:
            output_strs_df.to_csv(file, header=header, index=False)
    else:
        with open(model_pred_newfiles_path, 'a', newline='') as file:
            output_strs_df.to_csv(file, header=False, index=False)