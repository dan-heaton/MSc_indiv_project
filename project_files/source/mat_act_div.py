#Note: for this to work, we must *first* download the google sheets and place in the 'NSAA' file directory that is
#local to the user. Hence, the user must first download the google sheet at:
#'https://docs.google.com/spreadsheets/d/1OvkGU6kwmMxD6zdZqXcNKUvur1uFbAx5IND7_dXibjE', place it in the location of
#their NSAA directory, and change the global variable 'source_dir' below to reflect the path to their 'NSAA' directory

import argparse
import pandas as pd
import numpy as np
import sys
import os
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument("version", help="Specify the version that we wish to consider (i.e. the visit number of the "
                                    "patients). Must be either 'V1' or 'V2'.")
parser.add_argument("fn", help="Specify the patient ID to split the .mat file of. Specify 'all' for all files available.")
args = parser.parse_args()

source_dir = "C:\\msc_project_files\\NSAA\\"



def extract_act_times():
    data = pd.read_excel(source_dir + "DMD Start_End Frames.xlsx")
    patient_ids = data["Patient"].values.tolist()
    patient_ids = list(zip(patient_ids, np.arange(1, len(patient_ids)+1)))

    patient_ids_v1v2 = [[]]
    for pair in patient_ids:
        if pair[0] != "V2":
            patient_ids_v1v2[-1].append(pair)
        else:
            patient_ids_v1v2.append([])

    if args.version == "V1" or args.version == "V2":
        v1_v2_index = 0 if args.version == "V1" else 1
        id_pairs = patient_ids_v1v2[v1_v2_index]
        #Remove 'nan' pairs
        id_pairs = [id_pair for id_pair in id_pairs if str(id_pair[0]) != "nan"]
        ids = [id[0] for id in id_pairs]
        if args.fn in ids:
            #Retrieves the relevant row from the table and only includes the columns containing the
            act_times = data.iloc[id_pairs[ids.index(args.fn)][1]-1, 2:36].values
            #'Pair up' each start and end time for each activity within 'act_times'
            act_times_outer = [[[act_times[i], act_times[i+1]] for i in range(0, len(act_times), 2)]]
            return act_times_outer, ids
        elif args.fn == "all":
            act_times_outer = []
            for id in ids:
                act_times = data.iloc[id_pairs[ids.index(id)][1] - 1, 2:36].values
                act_times_outer.append([[act_times[i], act_times[i+1]] for i in range(0, len(act_times), 2)])
            return act_times_outer, ids
        else:
            print("Second arg ('fn') must be a patient ID within the specified version heading.")
            sys.exit()
    else:
        print("First arg ('version') must be one of 'V1' or 'V2'.")
        sys.exit()


def divide_mat_file(act_times_outer, ids):
    #Filters the file names within 'NSAA\matfiles\' by the chosen version
    global source_dir
    source_dir += "matfiles\\"
    if args.version == "V1":
        version_fns = [fn for fn in os.listdir(source_dir) if not fn.endswith("2.mat")
                       and not fn.endswith("3.mat") and "V2" not in fn]
    else:
        version_fns = [fn for fn in os.listdir(source_dir) if fn.endswith("2.mat") or "V2" in fn]

    if args.fn != "all":
        version_fns = [fn for fn in version_fns if args.fn in fn]
    else:
        version_fns = [fn for fn in version_fns if fn != "act_files"]

    if not os.path.exists(source_dir + "act_files\\"):
        os.mkdir(source_dir + "act_files\\")
    else:
        for fn in os.listdir(source_dir + "act_files\\"):
            os.remove(source_dir + "act_files\\" + fn)
    for file_name in version_fns:
        try:
            if len(act_times_outer) == 1:
                act_times = act_times_outer[0]
            else:
                act_times = act_times_outer[ids.index(file_name.split("-")[0])]
        except ValueError:
            print("Skipping " + file_name + " as not a row in table for version " + args.version + "...")
            continue

        #For a given patient name, if it doesn't have the necessary times in their respective row in the table already
        #filled in with numbers, skip it and continue to the next...
        exit_loop = 0
        for act in act_times:
            for a in act:
                if "nan" in str(a) or "-1" in str(a):
                    exit_loop = 1
        if exit_loop == 1:
            print("Skipping " + file_name + " as missing values in table for corresponding entry for version " +
                  args.version + "...")
            continue

        print("Dividing up " + file_name + "...")
        full_file_name = source_dir + file_name
        file = sio.loadmat(full_file_name)

        for i, pair in enumerate(act_times):
            inner_table = file["tree"][0][0][6][0][0][10][0][0][2][0]
            new_inner_table = inner_table[pair[0]:pair[1]]
            np.delete(file["tree"][0][0][6][0][0][10][0][0][2], 0)
            file["tree"][0][0][6][0][0][10][0][0][2] = [new_inner_table]
            new_file_name = str(file_name.split(".mat")[0]) + "_act" + str(i+1) + ".mat"
            sio.savemat(source_dir + "act_files\\" + new_file_name, file)
            file["tree"][0][0][6][0][0][10][0][0][2] = [inner_table]



act_times, ids = extract_act_times()
divide_mat_file(act_times, ids)
