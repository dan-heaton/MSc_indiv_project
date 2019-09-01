from settings import local_dir, sub_dirs
import argparse
import shutil
import sys
import os


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specify this as the directory within 'local_dir' in which to place the file.")
parser.add_argument("file_path", help="The path to the file in question which the user wishes to move to 'local_dir'. "
                                      "Either specify the local path from 'source' or the complete path.")
args = parser.parse_args()

#Ensures that the 'dir' argument is one of the allowed values
if args.dir + "\\" in sub_dirs:
    dir = args.dir
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'NMB', or 'direct_csv'.")
    sys.exit()

#Modifies the local copy of 'local_dir' to point to the correct subdirectory of the selected directory within
#'local_dir', dependent on the directory itself
if dir == "6minwalk-matfiles":
    local_dir_copy = local_dir + "6minwalk-matfiles\\all_data_mat_files\\"
elif dir == "NSAA":
    local_dir_copy = local_dir + "NSAA\\matfiles\\"
else:
    local_dir_copy = local_dir + dir

#If the user does not have the 'local directory' setup, creates one for them to contain this sole file so it can be
#correctly accessed by the data pipeline
if not os.path.exists(local_dir_copy):
    print("Creating the '" + local_dir_copy + "' directory to contain '" + args.file_path + "'...")
    os.makedirs(local_dir_copy)
if not os.path.exists(local_dir + "output_files"):
    print("Creating the output directory within '" + local_dir + "'...")
    os.makedirs(local_dir + "output_files")


#Attempts to copy the chosen file (given it's complete or relative path) to the chosen subdirectory of 'local_dir'
try:
    shutil.copy(args.file_path, local_dir_copy)
except FileNotFoundError:
    print("'" + args.file_path + "' not found as a source file...")
    sys.exit()