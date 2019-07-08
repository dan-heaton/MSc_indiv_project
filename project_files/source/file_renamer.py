import argparse
import os
import sys
from re import sub
from settings import local_dir, sub_dirs


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies the directory that we wish to use to source the file names that we wish to "
                                "rename. Must be one of '6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'allmatfiles'.")
args = parser.parse_args()

dir = args.dir



#Checks that the 'dir' argument is one of the allowed values and exits if it isn't
if dir + "\\" not in sub_dirs:
    print("First arg ('dir') must the name of the source directory from which we wish to transform the file names and "
          "must be one of '6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'allmatfiles'.")
    sys.exit()

#Gets the directory within 'dir' that contains the files of which we wish to change the name
if dir == "NSAA":
    local_dir += dir + "\\matfiles\\"
elif dir == "allmatfiles" or dir == "6MW-matFiles":
    local_dir += dir + "\\"
else:
    local_dir += dir + "\\all_data_mat_files\\"

#Filters out the non-mat files from the file names (i.e. so we only change the names of .mat files in 'local_dir)
file_names = [f for f in os.listdir(local_dir) if f.endswith(".mat")]

#Removes certain files within 'allmatfiles' that we don't want to keep, including any containing 'AllTasks' or 'Session'
files_to_delete, files_kept = [], []
if dir == "allmatfiles":
    for f in file_names:
        if "jointangle" not in f or "AllTasks" in f or "Session" in f:
            files_to_delete.append(f)
        else:
            files_kept.append(f)


#Creates new 'target' file names for each of the old file names based on sets of regexes for each 'dir'
new_file_names = []
if dir == "NSAA":
    for fn in files_kept:
        #Capitalizes files that start with 'd' to start with 'D'
        f = sub("^d", "D", fn)
        #Capitalizes the 'NSAA' part of the file
        f = sub("(nsaa|nsaa1|NSA1|NSA|NSAA|NSAAX)\.mat", "NSAA1.mat", f)
        #Captures the 'nsaa1a', 'nsaa1b', 'nsaa2a', and 'nsaa2b' cases
        f = sub("nsaa", "NSAA", f)
        #Gets rid of any 'xxx' within a 'NSAA' sequence
        f = sub("xxx", "1", f)
        #Handles the misspelled 'HC7' and 'HC3' files
        f = sub("7l", "7", f)
        f = sub("03", "3", f)
        f = sub("(nsaa2|NSA2)", "NSAA2", f)
        # Capitalizes any 'version' nums (e.g. 'Dv2' goes to 'DV2')
        f = sub("v", "V", f)
        new_file_names.append(f)
elif dir == "6minwalk-matfiles":
    for fn in files_kept:
        f = sub("All-", "", fn)
        f = sub("^d", "D", f)
        new_file_names.append(f)
elif dir =="6MW-matFiles":
    for fn in files_kept:
        f = sub("All-", "", fn)
        f = sub("^d", "D", f)
        #Removes extra string parts after '6MinWalk' or '6MW' (e.g. '6MinWalk-SensorDropped.mat' becomes '6MinWalk.mat')
        f = sub("6MinWalk.*\.mat", "6MinWalk.mat", f)
        f = sub("6MWT.*\.mat", "6MW.mat", f)
        new_file_names.append(f)
else:
    for fn in files_kept:
        #File names containing 'jointangle' has the beginning of each chopped off if it doesn't start with 'jointangle'
        f = sub(".*jointangle", "jointangle", fn)
        #Capitalize the beginning of short file names within the file name
        f = sub("jointangled", "jointangleD", f)
        #If the file is a 6 min walk file, make sure it ends with '6MinWalk.mat' and not '6MW.mat'
        f = sub("6MW\.mat", "6MinWalk.mat", f)
        #Replaces spaces with dashes (e.g. 'HC10 (15)' becomes 'HC10-(15)'
        f = sub(" ", "-", f)
        #Removes enclosing parenthesis around numbers (e.g. 'HC10-(15)' becomes 'HC10-15')
        f = sub("[()]", "", f)
        #Removes dashes after 'HC' and before the number of the short file name (e.g. 'HC-6' becomes 'HC6')
        f = sub("HC-", "HC", f)
        #Adds '.mat' to any files that are missing it
        f = sub("k$", "k.mat", f)
        new_file_names.append(f)


#Removes any of the files within 'allmatfiles' that we don't want to keep (i.e. discards 'AllTasks' or 'New-Session' files)
if files_to_delete:
    for file in files_to_delete:
        os.remove(local_dir + file)


#Takes the new file names that have been created in the predefined format and replaces the original file names with them
for nfn, fn in zip(new_file_names, files_kept):
    os.rename(local_dir + fn, local_dir + nfn)

