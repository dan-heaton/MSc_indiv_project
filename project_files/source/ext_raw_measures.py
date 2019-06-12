import argparse
import sys
import os
import scipy.io as sio
import pandas as pd

#Note: CHANGE THESE to location of the 3 sub-directories' encompassing directory local to the user
source_dir = "C:\\msc_project_files\\"
sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\"]
#Note: 'orientation' and 'sensorOrientation' are NOT included due to having '92' and '68' dimensions, respectively
#and not being in form of values for x,y,z dim
measurements = ["position", "velocity", "acceleration", "angularVelocity", "angularAcceleration",
                "sensorFreeAcceleration", "sensorMagneticField", "jointAngle", "jointAngleXZY"]

axis_labels = ["X", "Y", "Z"]

#Below 3 lists are labels for the 23 segments, 22 joints, and 17 sensors, as dictated by the 'MVN User Manual'
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


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory to use so as to process the files contained within "
                                "them accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles' or 'NSAA'.")
parser.add_argument("fn", help="Specifies the short name (e.g. 'D11') of the file that we wish to extract the specified "
                               "raw measurements. Specify 'all' for all the files available in the 'source_dir'.")
parser.add_argument("measurements", help="Specifies the measurements to extract from the source .mat file. Separate "
                                         "each measurement to extract by a comma.")
parser.add_argument("--single_act", type=bool, nargs="?", const=True,
                    help="Specify if the files to operate on are 'single act' files.")
args = parser.parse_args()


if args.dir + "\\" in sub_dirs:
    if args.dir == "6minwalk-matfiles":
        source_dir += args.dir + "\\all_data_mat_files\\"
    elif args.dir == "6MW-matFiles":
        source_dir += args.dir + "\\"
    else:
        source_dir += args.dir + "\\matfiles\\"
        if args.single_act:
            source_dir += "act_files\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'direct_csv'.")
    sys.exit()

file_names = os.listdir(source_dir)
if any(args.fn in fn for fn in file_names):
    full_file_names = [source_dir + [fn for fn in file_names if args.fn in fn][0]]
elif args.fn == "all":
    full_file_names = [source_dir + file_name for file_name in file_names if file_name.endswith(".mat")]
else:
    print("Second arg ('fn') must be the short name of a file (e.g. 'D2' or 'all') within", source_dir)
    sys.exit()

measures = []
if args.measurements == "all":
    measures = measurements
else:
    for measure in args.measurements.split(","):
        if measure in measurements:
            measures.append(measure)
        else:
            print("'" + measure + "' not a valid 'measurement' name. Must be 'all' or one of:", measurements)
            sys.exit()

for measure in measures:
    if not os.path.exists(source_dir + measure):
        os.mkdir(source_dir + measure)
    else:
        for f in os.listdir(source_dir + measure):
            os.remove(source_dir + measure + "\\" + f)

for full_file_name in full_file_names:
    print("\nExtracting", measures, "from '" + full_file_name + "'...")
    mat_file = sio.loadmat(full_file_name)
    tree = mat_file["tree"]
    try:
        frame_data = tree[0][0][6][0][0][10][0][0][3][0]
    except IndexError:
        frame_data = tree[0][0][6][0][0][10][0][0][2][0]
    col_names = frame_data.dtype.names
    # Extract single outer-list wrapping for vectors and double outer-list values for single values
    try:
        frame_data = [[elem[0] if len(elem[0]) != 1 else elem[0][0] for elem in row] for row in frame_data]
    except IndexError:
        # Accounts for missing 'contact' values in certain rows of some '.mat' files by ignoring the 'contact' column
        new_frame_data = []
        for m, row in enumerate(frame_data):
            # Ignore rows that don't have 'normal' as their 'type' cell
            if row[3][0] != "normal":
                continue
            row_data = []
            for i in range(len(row)):
                if i == len(row) - 1:
                    row_data.append(["", ""])
                elif len(row[i][0]) != 1:
                    row_data.append(row[i][0])
                else:
                    row_data.append(row[i][0][0])
            new_frame_data.append(row_data)
        frame_data = new_frame_data
    if not args.single_act:
        df = pd.DataFrame(frame_data, columns=col_names).iloc[3:]
    else:
        df = pd.DataFrame(frame_data, columns=col_names).iloc[:]

    for measure in measures:
        measurement_names = seg_join_sens_map[measure_to_len_map[measure]]
        headers = ["(" + measurement_name + ") : (" + axis + "-axis)"
                   for measurement_name in measurement_names for axis in axis_labels]

        measure_data = [list(data) for data in df.loc[:, measure].values]
        short_file_name = full_file_name.split("\\")[-1].split("-")[0]
        measure_df = pd.DataFrame(measure_data, index=[short_file_name for i in range(len(measure_data))])

        new_file_name = source_dir + measure + "\\" + full_file_name.split("\\")[-1].split(".mat")[0] + \
                        "_" + measure + ".csv"
        print("Writing '" + new_file_name + "' to '" + source_dir + measure + "\\'")
        measure_df.to_csv(new_file_name, header=headers)
