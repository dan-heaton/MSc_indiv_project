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

"""Section below encompasses all the required arguments for the script to run. Note there are only 2 arguments and they 
are BOTH required."""
parser = argparse.ArgumentParser()
parser.add_argument("version", help="Specify the version that we wish to consider (i.e. the visit number of the "
                                    "patients). Must be either 'V1' or 'V2'. Consult the google sheet and the relevant "
                                    "row subcategory the subject name is under for the correct version number.")
parser.add_argument("fn", help="Specify the patient ID to split the .mat file of. Specify 'all' for all files available.")
args = parser.parse_args()

#Note: CHANGE THIS to the location of the NSAA subdirectory local to the user's downloaded .mat files
source_dir = "C:\\msc_project_files\\NSAA\\"



def extract_act_times():
    """
    :return: list of start and end times of activities, as determined by the downloaded Google sheet (these are of a
    shape '# of files we are concerned with' x '# of total activities (i.e. 17)' x 2 (first being start time and
    second being end time); also returns a complete list of all files of the sheet under the subheading given by 'version'
    """
    #Reads in the downloaded Google sheet and makes a list of tuples, where each element in the list is a tuple of two
    #values: the name of an 'ID' cell and its index location within a different list of IDs
    data = pd.read_excel(source_dir + "DMD Start_End Frames.xlsx")
    patient_ids = data["Patient"].values.tolist()
    patient_ids = list(zip(patient_ids, np.arange(1, len(patient_ids)+1)))

    #Modifies the above list so that the tuples are now divided into two sub-lists of a list, with each being one of
    #the corresponding 'versions'
    patient_ids_v1v2 = [[]]
    for pair in patient_ids:
        if pair[0] != "V2":
            patient_ids_v1v2[-1].append(pair)
        else:
            patient_ids_v1v2.append([])

    if args.version == "V1" or args.version == "V2":
        v1_v2_index = 0 if args.version == "V1" else 1
        #Selects the relevant sublist of 'patient_ids_v1v2' depending on the version specified by the user
        id_pairs = patient_ids_v1v2[v1_v2_index]
        #Remove 'nan' pairs (i.e. rows of the spreadsheet with no patient entries but that are between several others)
        id_pairs = [id_pair for id_pair in id_pairs if str(id_pair[0]) != "nan"]
        #Gets a list of the first elements of the pairs, i.e. the new list contains just the ID names for the
        #given version
        ids = [id[0] for id in id_pairs]
        if args.fn in ids:
            #Retrieves the relevant row from the table and only includes the columns containing the activity times
            act_times = data.iloc[id_pairs[ids.index(args.fn)][1]-1, 2:36].values
            #'Pair up' each start and end time for each activity within 'act_times'; note that it is wrapped in an
            #outer list to make it the same type of list as would be returned if the 'fn' argument was 'all'
            act_times_outer = [[[act_times[i], act_times[i+1]] for i in range(0, len(act_times), 2)]]
            return act_times_outer, ids
        elif args.fn == "all":
            act_times_outer = []
            #Repeats the process outlined in the previous 'if' section but for every ID name in the previously
            #specified list
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
    """
    :param 'act_times_outer', which contains the list of start and end times of activites for all required files,
    and a list of the ID names for the given version, both as returned by the 'extract_act_times' argument,
    :return: no return, but instead divides up each relevant file by the file times specified in the 'act_times_outer'
    argument (i.e. for a given file name and a sequence of start and end times, chops up the .mat file to contain
    the data from the original of the time period specified by start and end time pairs)
    """

    #Appends the name of the relevant subdirectory within the 'NSAA' directory to correctly source the .mat files
    global source_dir
    source_dir += "matfiles\\"

    #Filters the file names within 'NSAA\matfiles\' by the chosen version
    if args.version == "V1":
        version_fns = [fn for fn in os.listdir(source_dir) if not fn.endswith("2.mat")
                       and not fn.endswith("3.mat") and "V2" not in fn]
    else:
        version_fns = [fn for fn in os.listdir(source_dir) if fn.endswith("2.mat") or "V2" in fn]

    #Filters down the list of files to split to either a list with one file name if 'fn' is a single name or a list
    #of all .mat files within 'source_dir' to use if 'fn' is 'all'
    if args.fn != "all":
        version_fns = [fn for fn in version_fns if args.fn in fn]
    else:
        version_fns = [fn for fn in version_fns if fn.endswith(".mat")]

    #Make the relevant subdirectory within 'source_dir' to host the divided-up act files or, if it already exists,
    #then remove all currently-held .mat files within 'act_files'
    if not os.path.exists(source_dir + "act_files\\"):
        os.mkdir(source_dir + "act_files\\")
    else:
        for fn in os.listdir(source_dir + "act_files\\"):
            if fn.endswith(".mat"):
                os.remove(source_dir + "act_files\\" + fn)

    #For each of the files we wish to split (either just one if 'fn' is single name or numerous if 'fn' is 'all')...
    for file_name in version_fns:
        #Ensure that, for a given file name, ensure there is a complete entry within the Google sheets table that
        #has the act times (as it needs this to divide up the file); if one doesn't exist, skip this file and move
        #onto the next one. Also, use the file name itself to determine the list of tuples to use as the start/end times
        try:
            if len(act_times_outer) == 1:
                act_times = act_times_outer[0]
            else:
                act_times = act_times_outer[ids.index(file_name.split("-")[0])]
        except ValueError:
            print("Skipping " + file_name + " as not a row in table for version " + args.version + "...")
            continue

        #For a given patient name, even if it has a corresponding row in the time (as determined by the previous
        #'try-catch' clause, if it doesn't have the necessary times in their respective row in the table already
        #filled in with numbers, skip it and continue on to the next
        exit_loop = 0
        for act in act_times:
            for a in act:
                if "nan" in str(a) or "-1" in str(a):
                    exit_loop = 1
        if exit_loop == 1:
            print("Skipping " + file_name + " as missing values in table for corresponding entry for version " +
                  args.version + "...")
            continue

        #Load the .mat file in question
        print("Dividing up " + file_name + "...")
        file = sio.loadmat(source_dir + file_name)

        #For each activity tuple of a start and end time, extract the table of data within the .mat file, take the rows
        #that are to be extracted (i.e. determined by the start- and end-time values within 'pair'), replaces the
        #original table of data within the .mat file with the shorter extracted one, save it within 'act_files' with
        #a name reflecting its source file and what activity it represents, and then resets the original source .mat file
        #for use within the next iteration of the 'for' loop
        for i, pair in enumerate(act_times):
            inner_table = file["tree"][0][0][6][0][0][10][0][0][2][0]
            new_inner_table = inner_table[pair[0]:pair[1]]
            np.delete(file["tree"][0][0][6][0][0][10][0][0][2], 0)
            file["tree"][0][0][6][0][0][10][0][0][2] = [new_inner_table]
            new_file_name = str(file_name.split(".mat")[0]) + "_act" + str(i+1) + ".mat"
            sio.savemat(source_dir + "act_files\\" + new_file_name, file)
            file["tree"][0][0][6][0][0][10][0][0][2] = [inner_table]


#Runs the script with the above-described two functions
act_times, ids = extract_act_times()
divide_mat_file(act_times, ids)
