#This file is only for the storing of variable names that are used throughout 'source' and variables are simply
#imported from here, rather than repeatedly re-defining things like 'local_dir' throughout the script. It also means
#there is only one point needed to change the variables in the directory rather than every script that uses it.


#Note: CHANGE THIS to location of the 3 sub-directories' encompassing directory local to the user.
local_dir = "C:\\msc_project_files\\"

#Other locations and sub-dir nameswithin 'local_dir' that will contain the files we need, as dictated by assuming the
#user has previously run the required scripts (e.g. 'comp_stat_vals', 'ext_raw_measures', etc.)
source_dir = local_dir + "output_files\\"
output_dir = local_dir + "output_files\\RNN_outputs\\"
model_dir = local_dir + "output_files\\rnn_models\\"
sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\", "direct_csv\\", "allmatfiles\\", "left-out\\"]
sub_sub_dirs = ["AD\\", "JA\\", "DC\\"]

#Other paths to documentation files that are referenced in several locations within 'source'
model_pred_path = "..\\documentation\\model_predictions.csv"
results_path = "..\\documentation\\RNN Results.xlsx"

#Types of files that are used to train models on
file_types = ["AD", "position", "velocity", "acceleration", "angularVelocity", "angularAcceleration",
                "sensorFreeAcceleration", "sensorMagneticField", "jointAngle", "jointAngleXZY"]
#The types of 'output' that models are trained towards
output_types = ["acts", "dhc", "overall"]




"""Section below includes names of parts of data that are defined outside the scope of this project (i.e. defined 
as part of the bodysuit equipment and that are given to us as part of the bodysuit user manual)"""

#List of possible measurements to extract from the source .mat files. Note that 'orientation' and 'sensorOrientation'
#are NOT included due to having '92' and '68' dimensions, respectively and not being in form of values for x,y,z dims
raw_measurements = ["position", "velocity", "acceleration", "angularVelocity", "angularAcceleration",
                "sensorFreeAcceleration", "sensorMagneticField", "jointAngle", "jointAngleXZY"]

axis_labels = ["X", "Y", "Z"]
short_file_types = ["JA", "AD", "DC"]


#Below 3 lists are labels for the 23 segments, 22 joints, and 17 sensors, respectively, as dictated by the 'MVN User Manual'
segment_labels = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head", "RightShoulder", "RightUpperArm",
                  "RightForeArm", "RightHand", "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
                  "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe", "LeftUpperLeg", "LeftLowerLeg",
                  "LeftFoot", "LeftToe"]
joint_labels = ["jL5S1", "jL4L3", "jL1T12", "jT9T8", "jT1C7", "jC1Head", "jRightT4Shoulder", "jRightShoulder",
                "jRightElbow", "jRightWrist", "jLeftT4Shoulder", "jLeftShoulder", "jLeftElbow", "jLeftWrist",
                "jRightHip", "jRightKnee", "jRightAnkle", "jRightBallFoot", "jLeftHip", "jLeftKnee",
                "jLeftAnkle", "jLeftBallFoot"]
sensor_labels = ["Pelvis", "T8", "Head", "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
                 "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand", "RightUpperLeg", "RightLowerLeg",
                 "RightFoot", "LeftUpperLeg", "LegLowerLeg", "LeftFoot"]


#Mapping used to map a given measurement name to the number of x/y/z values found in the .mat file
measure_to_len_map = {"orientation": 23, "position": 23, "velocity": 23, "acceleration": 23, "angularVelocity": 23,
                      "angularAcceleration": 23, "sensorFreeAcceleration": 17, "sensorMagneticField": 17,
                      "sensorOrientation": 22, "jointAngle": 22, "jointAngleXZY": 22}

#Mapping used to select lists of labels names to use based on the length of the numbers contained in data array
seg_join_sens_map = {len(segment_labels): segment_labels, len(joint_labels): joint_labels, len(sensor_labels): sensor_labels}


#Other random constants that are referenced in several points in 'source'
batch_size = 64
sampling_rate = 60      #In Hz