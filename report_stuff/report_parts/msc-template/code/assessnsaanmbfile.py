import os

file_path = input("\nPlease enter the path to the '.mat' file to assess (inc '.mat' extension): ")

nsaa_nmb = ""
while nsaa_nmb != "NSAA" and nsaa_nmb != "NMB":
    nsaa_nmb = input("\nPlease specify either 'NSAA' or 'NMB' for the type of activities contained in the file: ")


print("\n---------- Running 'file_mover.py...' ----------")
#Runs the 'file_mover.py' script to move the file to the 'NSAA' directory for
os.system("python file_mover.py " + nsaa_nmb + " " + file_path)

print("\n---------- Running 'file_renamer.py...' ----------")
#Runs the 'file_renamer.py' script to rename the file that has now been moved to the 'NSAA' directory
os.system("python file_renamer.py " + nsaa_nmb)


measures = input("\nPlease enter the measurements (separated by commas) to use to assess the file: ")


#Uses the full path name to the file to get the name of the file itself, the subject name within the file, and a
#list of measurements that we wish to extract as a list from the 'measures' arg
file_name = file_path.split("\\")[-1]
subject_name = file_name.split("-")[0]
measurements = measures.split(",")


print("\n---------- Running 'ext_raw_measures.py...' ----------")
#Extracts the raw measurements using all measurements specified apart from 'AD' (if it was included)
measures_no_ad = ",".join([m for m in measurements if m != "AD"])
os.system("python ext_raw_measures.py " + nsaa_nmb + " " + file_name + " " + measures_no_ad)

#Runs the 'comp_stat_vals.py' and 'ft_sel_red.py' scripts if 'AD' was one of the measurements given
if "AD" in measurements:
    print("\n---------- Running 'comp_stat_vals.py...' ----------")
    os.system("python comp_stat_vals.py " + nsaa_nmb + " AD " + file_name + " --split_size=1")
    print("\n---------- Running 'ft_sel_red.py...' ----------")
    os.system("python ft_sel_red.py " + nsaa_nmb + " AD " + subject_name +
              " pca --num_features=30 --no_normalize --new_subject --batch")


use_altdir = ""
while use_altdir != "y" and use_altdir != "n":
    use_altdir = input("\nDo you wish to assess the file on an 'alt dir' (e.g. if file is an 'NSAA' file, would "
                       "assess the file on models built only on NMB) ('y'/n')?: ")


print("\n---------- Running 'model_predictor.py...' ----------")
#Assess the models on the pre-built models contained within 'rnn_models_final' in the project directory
altdir = "NSAA" if nsaa_nmb == "NMB" else "NMB"
if use_altdir == "n":
    use_leaveoutV2 = ""
    while use_leaveoutV2 != "y" and use_leaveoutV2 != "n":
        use_leaveoutV2 = input("\nDo you wish to assess the file on models built that specifically haven't been "
                               "trained on any 'V2' files? ('y'/'n'): ")
    if use_leaveoutV2 == "n":
        os.system("python model_predictor.py " + nsaa_nmb + " " + measures + " " + subject_name + " --add_dir=" +
                  altdir + " --combine_preds --no_testset --use_seen --new_subject --final_models --no_nsaa_flag")
    else:
        os.system("python model_predictor.py " + nsaa_nmb + " " + measures + " " + subject_name + " --add_dir=" +
                  altdir + " --combine_preds --no_testset --use_seen --new_subject --final_models --no_nsaa_flag " +
                  "--leave_out_version=V2")
else:
    os.system("python model_predictor.py " + nsaa_nmb + " " + measures + " " + subject_name + "--alt_dirs=" +
              altdir + " --combine_preds --no_testset --use_seen --new_subject --final_models --no_nsaa_flag")