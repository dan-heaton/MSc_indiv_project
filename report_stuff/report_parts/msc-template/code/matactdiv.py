#Note: for this to work, we must *first* download the google sheets and place in the 'NSAA' file directory that is
#local to the user. Hence, the user must first download the google sheet at:
#'https://docs.google.com/spreadsheets/d/1OvkGU6kwmMxD6zdZqXcNKUvur1uFbAx5IND7_dXibjE', place it in the location of
#their NSAA directory, and change the global variable 'local_dir' below to reflect the path to their 'NSAA' directory

import argparse
import pandas as pd
import numpy as np
import sys
import os
import scipy.io as sio
from settings import local_dir, nsaa_subtasks_path

"""Section below encompasses all the required arguments for the script to run. Note there are only 2 arguments and they 
are BOTH required."""
parser = argparse.ArgumentParser()
parser.add_argument("fn", help="Specify the patient ID to split the .mat file of. Specify 'all' for all files available.")
parser.add_argument("--concat_act_files", type=bool, nargs="?", const=True, default=False,
                    help="Specify if wish to combine all the files together for a each subject (i.e. stack the tables "
                         "within the .mat files of each of the activities for a given subject).")
args = parser.parse_args()

local_dir += "NSAA\\"


def extract_act_times():
    """
    :return: list of start and end times of activities, as determined by the downloaded Google sheet (these are of a
    shape '# of subjects' x '# of total activities (i.e. 17)' x 3 (first being file name containing the activity,
    second being the start time and third being end time); also returns a complete list the subject names of which we
    are extracting the activities
    """
    #Reads in the downloaded Google sheet and makes a list of tuples, where each element in the list is a tuple of two
    #values: the name of an 'ID' cell and its index location within a different list of IDs
    data = pd.read_csv(nsaa_subtasks_path, delimiter=";")
    #Removes the final 12 columns loaded from the table, as these are concerning the '6-min walk' data
    data = data.iloc[:, :-12]
    #Drops the columns from the table that we don't need
    data = data.drop(labels=["AnnotatedBy", "GeneralNotes", "ReferenceVideo", "ReferenceVideoTime",
                             "RefMVNXFile", "ReferenceMVNXFrame"], axis=1)
    #Remove all the 'complete' and 'notes' columns in the table
    data = data.loc[:, ~(data == "completed").any()]
    data = data.loc[:, ~(data == "notes").any()]

    patient_ids = data["Patient"].values.tolist()
    id_pairs = list(zip(patient_ids, np.arange(0, len(patient_ids)+1)))
    #Removes the first pair in list, as this corresponds to row of table w/ 'filename', 'start', 'finish' etc.
    id_pairs = id_pairs[1:]


    #Gets a list of the first elements of the pairs, i.e. the new list contains just the ID names for the
    #given version
    ids = [id[0] for id in id_pairs]
    if args.fn in ids:
        #Retrieves the relevant row from the table and only includes the columns containing the activity times
        act_times = data.iloc[id_pairs[ids.index(args.fn)][1], 1:].values
        #'Pair up' each source file name, start time, and end time for each activity within 'act_times'; note that it
        #is wrapped in an outer list to make it the same type of list as would be returned if the 'fn' argument was 'all'
        act_times_outer = [[[act_times[i], act_times[i+1], act_times[i+2]] for i in range(0, len(act_times), 3)]]
        return act_times_outer, [args.fn]
    elif args.fn == "all":
        act_times_outer = []
        #Repeats the process outlined in the previous 'if' section but for every ID name in the previously
        #specified list
        for id in ids:
            act_times = data.iloc[id_pairs[ids.index(id)][1], 1:].values
            act_times_outer.append([[act_times[i], act_times[i+1], act_times[i+2]] for i in range(0, len(act_times), 3)])
        return act_times_outer, ids
    else:
        print("Second arg ('fn') must be a patient ID within the specified version heading.")
        sys.exit()



def divide_mat_file(act_times_outer, chosen_ids):
    """
    :param 'act_times_outer', which contains the list of start and end times of activites for all required files,
    and 'chosen_ids', which isa list of the subject ID names (list of one value if 'fn' is not 'all',
    else it's all the ID names)
    :return: no return, but instead divides up each subject's relevant files by the file times specified in the
    'act_times_outer' argument (i.e. for a given file name and a sequence of start and end times, chops up the
    .mat file to contain the data from the original of the time period specified by start and end time pairs)
    """

    #Appends the name of the relevant subdirectory within the 'NSAA' directory to correctly source the .mat files
    global local_dir
    local_dir += "matfiles\\"

    #Gets the list of files in the 'NSAA\matfiles' directory
    fns = [fn for fn in os.listdir(local_dir)]

    #Filters down the list of files to split to either a list with one file name if 'fn' is a single name or a list
    #of all .mat files within 'local_dir' to use if 'fn' is 'all'
    if args.fn != "all":
        fns = [fn for fn in fns if args.fn in fn]
    else:
        fns = [fn for fn in fns if fn.endswith(".mat")]

    if not args.concat_act_files:
        #Make the relevant subdirectory within 'local_dir' to host the divided-up act files or, if it already exists,
        #then remove all currently-held .mat files within 'act_files'
        if not os.path.exists(local_dir + "act_files\\"):
            os.mkdir(local_dir + "act_files\\")
        else:
            for fn in os.listdir(local_dir + "act_files\\"):
                if fn.endswith(".mat"):
                    os.remove(local_dir + "act_files\\" + fn)
    else:
        if not os.path.exists(local_dir + "act_files_concat\\"):
            os.mkdir(local_dir + "act_files_concat\\")
        else:
            for fn in os.listdir(local_dir + "act_files_concat\\"):
                if fn.endswith(".mat"):
                    os.remove(local_dir + "act_files_concat\\" + fn)



    #For each of the subjects we are extracting the activities of (either just one if 'fn' is single name or numerous if 'fn' is 'all')...
    for id, act_times in zip(chosen_ids, act_times_outer) :
        print("\nDividing up '" + id + "' activities...")

        mat_table = None
        mat_exists = False
        #For each of the 17 activities that the subject will have performed...
        for i, act in enumerate(act_times):
            #Finds first file beginning with the 'filename' for the activity and subject in question within 'local_dir'
            #If it's the second activity ('walking'), search for the file in the two 'walk' source directories
            walk_dirs = None
            if i == 1:
                #Removes any leading 'All-' from the file name
                fn = act[0].split("All-")[1] if act[0].startswith("All-") else act[0]
                #Capitalizes the leading 'd' if necessary
                fn = fn.split("-")[0].upper() + "-" + "-".join(fn.split("-")[1:])
                #Accounts for 'D11b' being 'D11B' instead
                fn = fn.split("-")[0][:-1] + "b-" + "-".join(fn.split("-")[1:]) if fn.split("-")[0][-1] == "B" else fn
                walk_dirs = ["\\".join(local_dir.split("\\")[:-3]) + "\\6MW-matFiles\\",
                             "\\".join(local_dir.split("\\")[:-3]) + "\\6minwalk-matfiles\\all_data_mat_files\\"]
                #Finds the file name in 'walk_dir' that exactly matches that of the 'filename' column for the
                #activity in the table (excluding variations of 'nsaa' in the file name, e.g. 'nsaa1')
                file_name = [[walk_dir + f for f in os.listdir(walk_dir) if f.split("-")[0] == fn.split("-")[0]]
                             for walk_dir in walk_dirs]
            else:
                # Capitalizes the leading 'd' if necessary
                fn = act[0].split("-")[0].upper() + "-" + "-".join(act[0].split("-")[1:])
                #Accounts for the 'HC7l' edge case
                fn = fn.split("-")[0][:-1] + "-" + "-".join(fn.split("-")[1:]) if "L" in fn.split("-")[0] else fn
                #Accounts for the 'HC03' instead of 'HC3'
                fn = fn.split("-")[0][:-2] + fn.split("-")[0][-1] + "-" + "-".join(fn.split("-")[1:]) if fn.split("-")[0][-2] == "0" else fn
                #Finds the file name in 'local_dir' that closely matches that of the 'filename' column for the
                #activity in the table
                file_name = [local_dir + f for f in os.listdir(local_dir) if fn.split("-")[0] + "-" in f.split("-")[0] + "-"
                             and fn.split("-")[1][-2:] in f.split("-")[1][-2:]]
            try:
                if i == 1:
                    #If the file name doesn't exist in either of the two walking directories, continue to next activity
                    if all(not walk_dir for walk_dir in walk_dirs):
                        raise IndexError
                    else:
                        #Assuming there's at least 1 walk file for the subject found in one of the directories,
                        #get the one from '6MW-matFiles', if it exists, otherwise get the one from '6minwalk-matfiles'
                        file_name = [file_name[0][0] if file_name[0] else file_name[1][0]]
                        file = sio.loadmat(file_name[0])
                else:
                    file = sio.loadmat(file_name[0])
                print("\tLoaded activity " + str(i+1) + " for subject '" + id + "' from file '" + file_name[0] + "'...")

            except IndexError:
                print("\tCannot extract activity " + str(i+1) + " for subject '" + id + "', as file '"
                      + act[0] + "' does not exist in relevant directory, continuing to next activity...")
                continue

            #Gets the start and end point of the activity to extract from within 'file', measured in rows
            start_time, end_time = int(act[1]), int(act[2])

            if start_time == -1 or end_time == -1:
                print("\tActivity contains '-1' in its 'start' and/or 'finish' times, meaning activity was not "
                      "performed by the subject...")
                continue


            #Reads in the table of data, cuts out the bit we want for this activity based on 'start_time' and
            #'end_time', replaces the old table with this one, and writes this new '.mat' file with this smaller table
            #to a new file based on the source file name and activity number
            try:
                inner_table = file["tree"][0][0][6][0][0][10][0][0][3][0]
                new_inner_table = inner_table[start_time:end_time]
                if len(new_inner_table) == 0:
                    print("\tEmpty activity " + str(i+1) + " (i.e. 'end_time' - 'start_time' = 0) in table, skipping...")
                    continue
                np.delete(file["tree"][0][0][6][0][0][10][0][0][3], 0)
                file["tree"][0][0][6][0][0][10][0][0][3] = [new_inner_table]
            except IndexError:
                inner_table = file["tree"][0][0][6][0][0][10][0][0][2][0]
                new_inner_table = inner_table[start_time:end_time]
                if len(new_inner_table) == 0:
                    print("\tEmpty activity " + str(i+1) + " (i.e. 'end_time' - 'start_time' = 0) in table, skipping...")
                    continue
                np.delete(file["tree"][0][0][6][0][0][10][0][0][2], 0)
                file["tree"][0][0][6][0][0][10][0][0][2] = [new_inner_table]
            if not args.concat_act_files:
                new_file_name = str(file_name[0].split("\\")[-1].split(".mat")[0]) + "_act" + str(i + 1) + ".mat"
                sio.savemat(local_dir + "act_files\\" + new_file_name, file)
                print("\tWritten " + new_file_name + " to '" + local_dir + "act_files\\'...")
            else:
                if not mat_exists:
                    mat_table = new_inner_table
                    mat_exists = True
                else:
                    try:
                        mat_table = np.append(mat_table, new_inner_table, axis=0)
                    #Accounts for rare problem w/ the combining of tables, specifically for 'D12' act 2
                    except TypeError:
                        print("\tCannot add activity '" + str(i+1) + "' for subject '" + id + "' due to 'TypeError'...")
                        continue
            file["tree"][0][0][6][0][0][10][0][0][2] = [inner_table]
        if args.concat_act_files:
            new_file_name = id + "_concat_acts.mat"
            #Loads a template file that we use to get the formating right, but whose table we replace with 'np.delete()'
            template_file = sio.loadmat(local_dir + [f for f in os.listdir(local_dir) if f.endswith(".mat")][0])
            # Deletes the templates data table and replaces it with the concatenated data
            np.delete(template_file["tree"][0][0][6][0][0][10][0][0][2], 0)
            template_file["tree"][0][0][6][0][0][10][0][0][2] = [mat_table]
            #Saves this concatenated data to a new file
            sio.savemat(local_dir + "act_files_concat\\" + new_file_name, template_file)
            print("\tWritten concat " + new_file_name + " to '" + local_dir + "act_files_concat\\'...")



act_times, chosen_ids = extract_act_times()
divide_mat_file(act_times, chosen_ids)
