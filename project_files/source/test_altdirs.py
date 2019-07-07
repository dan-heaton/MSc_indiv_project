import argparse
import sys
import os

"""Section below specifies the arguments that are required to run the script, which needs a directory from which to 
source the files to test, the directories that are used to train the models that we shall be testing on, and 
the names of the file types (i.e. measurements, be they raw measurements or 'AD' files) from the source directory that 
we are testing upon."""
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies the directory that we wish to use to source the testing files from. Must be "
                                "one of '6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'allmatfiles'.")
parser.add_argument("testdirs", help="Specifies the directories of files that are used to train the models used "
                                     "for file assessment. Must be one of '6minwalk-matfiles', '6MW-matFiles', "
                                     "'NSAA', or 'allmatfiles', and should be separated with '_'.")
parser.add_argument("ft", help="Specifies the measurements that will be used from the files in 'dir' to test on "
                               "the models sourced from 'testdirs'.")
args = parser.parse_args()



#Note: CHANGE THIS to location of the 3 sub-directories' encompassing directory local to the user.
local_dir = "C:\\msc_project_files\\"
source_dir = local_dir + "output_files\\"

sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\", "direct_csv\\", "allmatfiles\\"]
file_types = ["AD", "position", "velocity", "acceleration", "angularVelocity", "angularAcceleration",
                "sensorFreeAcceleration", "sensorMagneticField", "jointAngle", "jointAngleXZY"]



#Section below ensures that the arguments provided to the script are valid in the context of how we will be
#using them and exits out if they aren't allowed.

if args.dir + "\\" not in sub_dirs:
    print("First arg ('dir') must be one of '6minwalk-matfiles', 6MW-matFiles', 'NSAA', 'allmatfiles', or 'direct_csv'.")
    sys.exit()

test_dirs = args.testdirs.split("_")
for td in test_dirs:
    if td + "\\" not in sub_dirs:
        print("Second arg ('testdirs') must be one or more dirs separated by '_' and each must be one of "
              "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'allmatfiles', or 'direct_csv'.")

#Ensures the corresponding second argument when dealing with files from 'allmatfiles'
if args.dir == "allmatfiles" and args.ft != "jointAngle":
    print("Third arg ('ft') must be 'jointAngle when 'dir' argument is \'allmatfiles\'")
    sys.exit()

for ft in args.ft.split(","):
    if ft not in file_types:
        print("Third arg ('ft') must be comma separated file types and must be within 'file_types'.")
        sys.exit()


#Based on which 'dir' we are using, load the 'short names' of each of the '.mat' files that correspond to the full
#file names within the 'dir' directory

if args.dir == "allmatfiles":
    source_dir = local_dir + args.dir + "\\"
    #Note: the check for files not containing 'AllTasks' or 'ttest' is to prevent 'model_predictor' trying to predict
    #on those files since 'ext_raw_measures' doesn't extract measures from these
    short_file_names = [f.split("jointangle")[-1].split(".mat")[0].upper() for f in os.listdir(source_dir)
                        if f.endswith(".mat") and "AllTasks" not in f and "ttest" not in f and "Session" not in f]
elif args.dir == "6MW-matFiles":
    source_dir = local_dir + args.dir + "\\"
    short_file_names = [f.split("-")[1] for f in os.listdir(source_dir) if f.startswith("All") and f.endswith(".mat")]
    short_file_names += [f.split("-")[0] for f in os.listdir(source_dir) if not f.startswith("All") and f.endswith(".mat")]
elif args.dir == "NSAA":
    source_dir = local_dir + args.dir + "\\matfiles\\"
    short_file_names = [f.upper().split("-")[0] for f in os.listdir(source_dir) if f.endswith(".mat")]
else:
    source_dir = local_dir + args.dir + "\\all_data_mat_files\\"
    short_file_names = [f.split("-")[1] for f in os.listdir(source_dir) if f.endswith(".mat")]




len_sfn = len(short_file_names)

#For each of the short file names that are contained within 'dir', run 'model_predictor.py' for every short file name
#and with arguments specified by the arguments given to 'test_altdirs'
for i, sfn in enumerate(short_file_names):
    print("\nFile: " + str(i+1) + " / " + str(len(short_file_names)) + ": " + sfn + "\n")
    if sfn.startswith("-"):
        mod_pred_str = "python model_predictor.py " + args.dir + " " + args.ft + " " + sfn[1:] \
                       + " --alt_dirs=" + args.testdirs + " --handle_dash --file_num=" \
                       + str(i+1) + " / " + str(len_sfn)
    else:
        mod_pred_str = "python model_predictor.py " + args.dir + " " + args.ft + " " + sfn + \
                       " --alt_dirs=" + args.testdirs + " --file_num=" \
                       + str(i+1) + "/" + str(len_sfn)
    os.system(mod_pred_str)
