import sys
sys.path.append("..")
import scipy.io as sio
import numpy as np
from numpy import linalg as la
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from tkinter import*
from os.path import isfile
import argparse

sampling_rate = 60      #In Hz

#Note: CHANGE THESE to location of '6minwalk-matfiles' directory local to the user
source_dir = "C:\\"
output_dir = "output_files\\"

file_types = ["ja", "ad", "dc"]
ja_dir = source_dir + "6minwalk-matfiles\\joint_angles_only_matfiles\\"
ja_short_file_names = ["D2", "D3", "D4", "HC2", "HC5", "HC6"]
ad_dir = source_dir + "6minwalk-matfiles\\all_data_mat_files\\"
#Note: complete file names for 'd5' and 'HC9' have been changed to be the same format as the others;
#worth doing so for the user to avoid inevitable FileNotFoundErrors that would otherwise appear
ad_short_file_names = ["D2", "D3", "d4", "d5", "D6", "D7", "HC1", "HC2", "HC3",
                       "HC4", "HC5", "HC7", "HC8", "HC9", "HC10"]
axis_labels = ["X", "Y", "Z"]


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
seg_join_sens_map = {len(segment_labels): segment_labels,
                     len(joint_labels): joint_labels,
                     len(sensor_labels): sensor_labels}


"""
Section below covers a few general-purpose mathematical functions used for several file object types 
for statistical analysis
"""
def mean_round(nums):
    """
        :param list of values of which we wish to find the mean/average:
        :return a float value rounded to 2 decimal places of the mean of 'nums':
    """
    return round(float(np.mean(nums)), 4)

def variance_round(nums):
    """
        :param list of values of which we wish to find the variance:
        :return a float value rounded to 2 decimal places of the variance of 'nums':
    """
    return round(float(np.var(nums)), 4)

def compute_diffs(nums):
    """
        :param list of values of which we wish to find the diffs between each successive value:
        :return list of diff values of len = len(nums)-1, where each value is difference between value
        in the 'nums' list and the next value in the list:
    """
    return [(nums[i+1]-nums[i]) for i in range(len(nums)-1)]

def mean_diff_round(nums):
    """
        :param list of values of which we wish to find the mean diff values:
        :return mean value of the absolute values of the diffs list (see 'compute_diffs'):
    """
    return round(float(np.mean(np.absolute(compute_diffs(nums)))), 4)

def fft_1d_round(nums):
    fft_1d = np.fft.rfft(nums, n=3)
    fft_1d.sort()
    return round(np.real(np.flip(fft_1d)[0]), 4)

def covariance_round(nums):
    """
        :param list of values of 2D array of numbers to find covariance of (shape = # of dimensions of data (e.g. 3 for
        x,y,z data) x # of samples):
        :return covariance values of rounded float values of shape = # of dimensions of data x # of dimensions of data,
        for 'top' right triangle at 1D list:
    """
    return [[round(float(n), 4) for n in num] for num in np.cov(nums.tolist())]

def fft_2d_round(nums):
    """
    :param list of 2d values of which we want to find the 2-dimensional FFT:
    :return result of 2D fft operation with shape given by 'shape' param and values rounded
    """
    fft_2d = np.fft.fft2(nums, s=(1, 3))
    fft_2d.sort()
    fft_2d = np.flip(fft_2d)
    return [round(np.real(fft_2d[0][i]), 4) for i in range(len(fft_2d[0]))]

def mean_sum_vals(nums):
    return round(float(np.mean([np.sum(nums[i]) for i in range(len(nums))])), 4)

def mean_sum_abs_vals(nums):
    return round(float(np.mean([np.sum(np.abs(nums[i])) for i in range(len(nums))])), 4)

def cov_eigenvals(nums, i, inner_i, inner_j):
    vals = la.eig(covariance_round(nums))[0].tolist()
    top_vals = [round(val, 4) for val in vals if val != min(vals)]
    top_vals.sort(reverse=True)
    return round(top_vals[i], 4)

def prop_outside_mean_zone(nums, percen=0.1):
    x_mean, y_mean, z_mean = np.mean(nums[0]), np.mean(nums[1]), np.mean(nums[2])
    nums_outside = 0
    nums = np.swapaxes(nums, 0, 1)
    for num in nums:
        if not x_mean*(1-percen) < num[0] < x_mean*(1+percen):
            if not y_mean*(1-percen) < num[1] < y_mean*(1+percen):
                if not z_mean*(1-percen) < num[2] < z_mean*(1+percen):
                    nums_outside += 1
    return round(nums_outside/len(nums), 4)




def extract_stats_features(f, split_file, file_name, measurement_names, file_part, is_recursive_call=False):
    data = {}
    # Adds a 'y' label for a given file
    data['file_type'] = "HC" if "HC" in file_name else "D"
    for i in range(len(file_part)):  # For each measurement category
        seg_join_sens_labels = seg_join_sens_map[measure_to_len_map[measurement_names[i]]]
        num_features = 1 if is_recursive_call else measure_to_len_map[measurement_names[i]]
        for j in range(num_features):  # For each feature (e.g. segment, joint, or sensor)
            for k in range(len(file_part[i][j])):  # For each x,y,z dimension
                if is_recursive_call:
                    stat_name = "(" + measurement_names[i] + ") : (Over all features) : (" + axis_labels[k] + "-axis)"
                else:
                    stat_name = "(" + measurement_names[i] + ") : (" + seg_join_sens_labels[j] + ") : (" + axis_labels[k] + "-axis)"
                data[stat_name + " : (mean)"] = mean_round(file_part[i][j][k])
                data[stat_name + " : (variance)"] = variance_round(file_part[i][j][k])
                data[stat_name + " : (abs mean sample diff)"] = mean_diff_round(file_part[i][j][k])
                data[stat_name + " : (FFT largest val)"] = fft_1d_round(file_part[i][j][k])
            if is_recursive_call:
                xyz_stat_name = "(" + measurement_names[i] + ") : (Over all features) : ((x,y,z)-axis) : "
            else:
                xyz_stat_name = "(" + measurement_names[i] + ") : ( " + seg_join_sens_labels[j] + ") : ((x,y,z)-axis) : "
            data[xyz_stat_name + "(mean sum vals)"] = mean_sum_vals(file_part[i][j])
            data[xyz_stat_name + "(mean sum abs vals)"] = mean_sum_abs_vals(file_part[i][j])
            data[xyz_stat_name + "(first eigen cov)"] = cov_eigenvals(file_part[i][j], 0, i, j)
            data[xyz_stat_name + "(second eigen cov)"] = cov_eigenvals(file_part[i][j], 1, i, j)
            data[xyz_stat_name + "(x- to y-axis covariance)"] = covariance_round(file_part[i][j])[0][1]
            data[xyz_stat_name + "(x- to z-axis covariance)"] = covariance_round(file_part[i][j])[0][2]
            data[xyz_stat_name + "(y- to z-axis covariance)"] = covariance_round(file_part[i][j])[1][2]
            data[xyz_stat_name + "(FFT 1st largest val)"] = fft_2d_round(file_part[i][j])[0]
            data[xyz_stat_name + "(FFT 2nd largest val)"] = fft_2d_round(file_part[i][j])[1]
            data[xyz_stat_name + "(FFT 3rd largest val)"] = fft_2d_round(file_part[i][j])[2]
            data[xyz_stat_name + "(proportion samples outside mean zone)"] = prop_outside_mean_zone(file_part[i][j])

    #Handles recursive case of function when computing statistics over all the measurements' features
    if len(file_part[0]) == 1:
        return data

    #Concatenates samples along the 'features' axis (i.e. so data now:
    # '# measurements' x 1 x '# axes' x '(# samples x # features)'
    new_file_part = np.reshape(file_part, (len(file_part), 1, len(file_part[0][0]), -1))
    data_over_all = extract_stats_features(f, split_file, file_name,
                                           measurement_names, new_file_part, is_recursive_call=True)
    data.update(data_over_all)

    # Creates and returns a DataFrame object from a single list version of the dictionary (so it's only 1 row),
    # and creates either a .csv with the default output_name (w/ .csv name from the AD short file name)
    # or with a given name and, if the given name already exists, appends it to the end of existing file
    return pd.DataFrame([data],columns=data.keys(),index=[file_name + "_(" + str(f + 1) + "/" + str(split_file) + ")"])



def check_for_abnormality(file_names, error_margin=1, abnormality_threshold=0.3):
    """
        :param 'file_names' to be the names of the files to check for abnormality, 'error_margin' to be the proportion
        of the mean of a feature to compute the upper and lower bound (e.g. error_margin=0.2 for a feature of mean
        '5' would mean a file part's value has to be within '4' and '6' for that feature to be considered a 'normal'
        value), and 'abnormality_threshold' to be the portion of features for a file part outside of the normal ranges
        for the file part to be considered 'abnormal'
        :return: no return, but prints the names of the file parts that are considered to be abnormal
    """
    for m in range(len(file_names)):
        file_df = pd.read_csv(file_names[m], index_col=0).iloc[:, 1:]
        # Note: only computes the means for features w/ single values (e.g. mean), not multi-dims (e.g. covariances)
        # for i in range(len(file_df.columns)):
        cols_to_keep = [i for i in range(len(file_df.values[0])) if type(file_df.values[0, i])==float]
        cols_to_remove = [i for i in range(len(file_df.values[0])) if i not in cols_to_keep]
        file_df = file_df.drop(file_df.columns[cols_to_remove], axis=1)
        col_means = [np.mean(file_df.iloc[:, i]) for i in range(len(file_df.iloc[0]))]

        for i in range(len(file_df)):
            features_out_of_range = 0
            for j in range(len(file_df.iloc[0])):
                feature, mean = abs(file_df.iloc[i, j]), abs(col_means[j])
                lb, ub = mean - mean*error_margin, mean + mean*error_margin
                if feature < lb or feature > ub:
                    features_out_of_range += 1
            if features_out_of_range / len(file_df.iloc[0]) > abnormality_threshold:
                print(file_df.index.values[i], "outside", error_margin, "mean error margin for >",
                      (abnormality_threshold*100), "% of its features")




class AllDataFile(object):

    def __init__(self, ad_file_name, fft_shape=(1, 10)):
        """
            :param string name of the unique identifier of an 'All-' data matlab file (e.g. 'HC3'), along with
            an optionally specified 2D shape to return from the FFT functionality applied to the measurements:
            :return no return, but sets up the DataFrame 'df' attribute from the given matlab file: name
        """
        self.ad_file_name = ad_file_name
        self.fft_shape = fft_shape
        # Loads the data from the given file name and extracts the root 'tree' from the file
        ad_data = sio.loadmat(ad_dir + "All-" + ad_file_name + "-6MinWalk")
        tree = ad_data["tree"]
        # Corresponds to location in matlabfile at: 'tree.subjects.frames.frame'
        #Try-except clause here to catch when 'D6' doesn't have a 'sensorCount' category within 'tree.subject.frames'
        print("Extracting data from AD file " + self.ad_file_name + "....")
        try:
            frame_data = tree[0][0][6][0][0][10][0][0][3][0]
        except IndexError:
            frame_data = tree[0][0][6][0][0][10][0][0][2][0]
        # Gets the types in each column of the matrix (i.e. dtypes of submatrices)
        col_names = frame_data.dtype.names
        # Extract single outer-list wrapping for vectors and double outer-list values for single values
        frame_data = [[elem[0] if len(elem[0]) != 1 else elem[0][0] for elem in row] for row in frame_data]
        df = pd.DataFrame(frame_data, columns=col_names)
        self.df = df


    def display_3d_positions(self):
        """
            :param df:
            :return no return, but plots position values for the given file object on a 3D dynamic plot,
            with the necessary components connected:
        """
        #Disregard first 3 samples (usually types 'identity', 'tpose' or 'tpose-isb')
        positions = self.df.loc[:, "position"].values[3:]
        #Extracts position values for each dimension so 'xyz_pos' now has shape:
        # # of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K) x # of features (23 for segments)
        xyz_pos = [[[s for s in sample[i::3]] for sample in positions] for i in range(3)]
        # # of features (23 for segments) x # of samples (~22K) x # of dimensions (e.g. 3 for x,y,z position values)
        xyz_pos_t = np.swapaxes(xyz_pos, 0, 2)
        stick_defines = [(0, 1), (0, 15), (0, 19), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                         (7, 8), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14), (15, 16),
                         (16, 17), (17, 18), (19, 20), (20, 21), (21, 22)]
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        maxs = [max([max(xyz_pos[i][j]) for j in range(len(xyz_pos[i]))]) for i in range(len(xyz_pos))]
        mins = [min([min(xyz_pos[i][j]) for j in range(len(xyz_pos[i]))]) for i in range(len(xyz_pos))]
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2, 1.5, 0.25, 1]))

        colors = plt.cm.jet(np.linspace(0, 1, len(xyz_pos[0][0])))
        lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
        pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

        stick_lines = [ax.plot([], [], [], 'k-')[0] for _ in stick_defines]

        def init():
            for line, pt in zip(lines, pts):
                line.set_data([], [])
                line.set_3d_properties([])
                pt.set_data([], [])
                pt.set_3d_properties([])
            return lines + pts + stick_lines

        def animate(i):
            if (i*5)%sampling_rate == 0:
                print("Plotting time: " + str((i*5)/sampling_rate) + "s")
            #print(i)
            i = (5 * i) % xyz_pos_t.shape[1]
            for line, pt, xi in zip(lines, pts, xyz_pos_t):
                x, y, z = xi[:i].T
                pt.set_data(x[-1:], y[-1:])
                pt.set_3d_properties(z[-1:])
            for stick_line, (sp, ep) in zip(stick_lines, stick_defines):
                stick_line._verts3d = xyz_pos_t[[sp, ep], i, :].T.tolist()
            fig.canvas.draw()
            return lines + pts + stick_lines

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xyz_pos[0]),
                                       interval=1000 / sampling_rate, blit=False, repeat=False)
        plt.show()



    def file_info(self):
        """
            :param string name of the unique identifier of an 'All-' data matlab file (e.g. 'HC3'):
            :return no return, but sets up the DataFrame 'df' attribute from the given matlab file: name
        """
        print("\nAd " + self.ad_file_name + " keys:", ", ".join([("'" + key + "'") for key in self.df]), "\n")
        for k in list(self.df.keys())[:3]:
            print(k, ":", (self.df[k]))


    def write_statistical_features(self, measurements_to_extract=None, output_name=None, split_file=1):
        """
            :param measurements_to_extract to be an optional list of measurement names from the AD file that we wish to
            apply statistical analysis (otherwise, defaults to 'measurement_names'), along with 'output_name' that
            defaults to the short name of the AD file, which writes to an individual file for the object (i.e.
            a single line .csv for the object); specify shared 'output_name' to instead append to an existing .csv:
            :return no return, but writes to an output .csv file the some of the statistical features of the certain
            measurements to an output .csv file
        """
        #Sets the default measurements to extract from the AD file object if none are specified as args
        measurement_names = measurements_to_extract if measurements_to_extract else [
            "position", "sensorFreeAcceleration", "sensorMagneticField", "velocity", "angularVelocity",
            "acceleration", "angularAcceleration"]

        # Extracts values of the various measurements in different layout so 'f_d_s_data' now has shape:
        # (# measurements (e.g. 3 for position, accel, and megnet field) x # of features (e.g. 23 for segments)
        # x # of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K))
        extract_data = self.df.loc[3:, measurement_names].values
        f_d_s_data = np.zeros((len(measurement_names),
                               max(len(segment_labels), len(joint_labels), len(sensor_labels)), 3,
                               len(self.df.loc[:])-3))
        for i in range(len(measurement_names)):
            for j in range(int(len(self.df.loc[3:, measurement_names[i]].values[0])/3)):
                for k in range(len(axis_labels)):
                    for m in range(len(self.df.loc[3:])):
                        f_d_s_data[i, j, k, m] = extract_data[m, i][(j*3)+k]

        split_size = int(len(f_d_s_data[0, 0, 0])/split_file)
        written_names = []
        for f in range(split_file):
            fds = f_d_s_data[:, :, :, (split_size*f):(split_size*(f+1))]
            df = extract_stats_features(f=f, split_file=split_file, file_name=self.ad_file_name,
                                        measurement_names=measurement_names, file_part=fds)
            if not output_name:
                output_complete_name = output_dir + "AD_" + self.ad_file_name + "_stats_features.csv"
                print("Writing AD", self.ad_file_name, "(", (f+1), "/", split_file, ") statistial info to",
                      output_complete_name)
            else:
                output_complete_name = output_dir + "AD_" + output_name + "_stats_features.csv"
                print("Writing AD", self.ad_file_name, "(", (f + 1), "/", split_file, ") statistial info to",
                      output_complete_name)
            if isfile(output_complete_name):
                with open(output_complete_name, 'a') as file:
                    df.to_csv(file, header=False)
            else:
                with open(output_complete_name, 'w') as file:
                    df.to_csv(file, header=True)
            written_names.append(output_complete_name)
        return written_names





class DataCubeFile(object):

    def __init__(self, dis_data_cube=False):
        """
            :param dis_data_cube is specified to True if wish to display basic Data Cube info to the screen:
            :return no return, but reads in the datacube object from .mat file, the datacube table from the .csv,
            extracts the names of the joint angle files in the datacube file, and instantiates JointAngleFile objects
            for each file contained in the datacube .mat file

            IMPORTANT NOTE: as there's no conceivable way currently to read a matlab table into Python (unlike structs),
            it's required that the matlab table is exported to a .csv before creating DataCubeFileObject. Hence, with
            the data_cube .mat file open in matlab, run writetable(excel_table, "data_cube_table.csv") in matlab for the
            following to work
        """

        try:
            self.dc_table = pd.read_csv(ja_dir + "data_cube_table.csv")
            self.dc_short_file_names = self.dc_table.values[:, 1]
        except FileNotFoundError:
            print("Couldn't find the 'data_cube_table.csv file. Make sure to run the "
                  "'writetable(excel_table, \"data_cube_table.csv\") in matlab with data_cube.mat file open")
            sys.exit()
        self.dc = sio.loadmat(ja_dir + "data_cube_6mw", matlab_compatible=True)["data_cube"][0][0][2][0]

        self.ja_objs = []
        for i in range(len(self.dc_short_file_names)):
            print("Extracting data from data cube file (" + str(i+1) + "/" + str(len(self.dc_short_file_names)) +
                  "): " + self.dc_short_file_names[i] + "...")
            self.ja_objs.append(JointAngleFile(self.dc_short_file_names[i], dc_object_index=i))

        if dis_data_cube:
            self.display_info()



    def display_info(self):
        """
            :param no params
            :return no return, but displays some basic info about the Data Cube attribute
        """
        print("\nData cube keys:", ", ".join([("'" + key + "'") for key in self.dc]), "\n")
        print("'__header__': ", self.dc["__header__"])
        print("'__version__': ", self.dc["__version__"])
        print("'__globals__': ", self.dc["__globals__"])
        print(self.dc["data_cube"])


    def write_statistical_features(self, output_name="all", split_file=1):
        """
        :param 'output_name', which defaults to 'All', which is the short name of the output .csv stats file that the
        objects will share...specify a different name if desired
        :return no return, but creates a single .csv output file that contains the statistical analysis of each joint
        angle file contained within the data cube via calls to JointAngleFile.write_statistical_features() method
        for each of the joint angle objects extracted at initialization
        """
        for i in range(len(self.ja_objs)):
            self.ja_objs[i].write_statistical_features(output_name=output_name, split_file=split_file)



class JointAngleFile(object):

    def __init__(self, ja_file_name, dc_object_index=-1):
        """
            :param string name of the unique identifier of an 'jointangle-' data matlab file (e.g. 'D2'); also includes
            'dc_object', which is set to non-zero if object is created as part of a DataCubeFile:
            :return no return, but extracts the data from the specified matlab file and extracts the features
            from each dimension
        """
        self.ja_file_name = ja_file_name
        self.is_dc_object = True if dc_object_index != -1 else False
        ja_p = "jointangle"
        ja_s = "-6MinWalk"

        #Loads the JA data from the datacube file instead of the normal JA files if passed from DataCubeFile class
        if dc_object_index != -1:
            self.ja_data = sio.loadmat(ja_dir + "data_cube_6mw",
                                  matlab_compatible=True)["data_cube"][0][0][2][0][dc_object_index]
        else:
            #'Try-except' clause included to handle some slight naming inconsistencies with the JA filename syntax
            try:
                self.ja_data = sio.loadmat(ja_dir + ja_p + ja_file_name + ja_s)['jointangle']
            except FileNotFoundError:
                self.ja_data = sio.loadmat(ja_dir + ja_p + ja_file_name + ja_s + "-TruncLen5Min20Sec")['jointangle']

        print("Extracting data from JA file " + self.ja_file_name + "....")
        #x/y/z_angles arrays have shape (# of samples in JA data file x # of features (i.e. # of features, 22))
        self.x_angles = np.asarray([[s for s in sample[0::3]] for sample in self.ja_data])
        self.y_angles = np.asarray([[s for s in sample[1::3]] for sample in self.ja_data])
        self.z_angles = np.asarray([[s for s in sample[2::3]] for sample in self.ja_data])
        #xyz_angles array has shape (# of samples in JA data file x # of features (i.e. # of features, 22)
        # x # of dimensions of data (i.e. 3 for x,y,z))
        self.xyz_angles = np.asarray([[[x, y, z] for x, y, z in
                                       zip([s for s in sample[0::3]], [s for s in sample[1::3]],
                                           [s for s in sample[2::3]])] for sample in self.ja_data])


    def display_3d_angles(self):
        """
            :param no params
            :return no return, but displays the angles of each feature in 3D as it changes over time (i.e. with sample num)
        """
        ja_3d_data = self.xyz_angles
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        for i, sample in enumerate(ja_3d_data):
            try:
                ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
                plt.pause((1/sampling_rate))
                ax.clear()
            except TclError:
                print("Window closed on frame:", (i+1))
                sys.exit()
        plt.close()


    def display_diffs_plot(self):
        """
            :param no params
            :return no return, but displays (# of features (e.g. 22 features) x # of dimensions (e.g. 3 for x,y,z))
            subplots for plot of diffs for each feature's dimension over time (e.g. 2,3 graph shows the diffs of
            successive joint angles values for the the 2nd feature's z-axis)
        """
        plt.rcParams.update({'font.size': 5})
        #Creates diff lists for all features and dimensions of the joint angle file object, with shape:
        #(# of dimensions (e.g. 3 for x,y,z) x # of features (e.g. 22 features) x # of samples - 1)
        x_y_z_diffs = np.asarray([[compute_diffs(self.xyz_angles[:, i, j]) for i in range(len(self.xyz_angles[0]))]
                                  for j in range(len(self.xyz_angles[0][0]))])

        i_dim = len(x_y_z_diffs)
        j_dim = len(x_y_z_diffs[0])
        #Makes sure all subplots share the same axes for easier comparison
        fig, axes = plt.subplots(j_dim, i_dim, sharex='all', sharey='all')
        for i in range(i_dim):  # For each x/y/z dimension
            for j in range(j_dim):  # For each of the joint angles
                #Uses 'time_increments' for x_axis points, with 1 increment (in 's') corresponding to one diff sample
                #for each subplot
                time_increments = np.arange(0, (len(x_y_z_diffs[i][j]) / sampling_rate), (1 / sampling_rate))
                axes[j, i].plot(time_increments, x_y_z_diffs[i][j])
                axes[j, i].set_ylabel(joint_labels[j] + ": " + axis_labels[i], rotation=0, size=6, labelpad=25)

        plt.subplots_adjust(hspace=0.1, top=0.95, bottom=0.03, right=0.99, left=0.07)
        plt.gcf().set_size_inches(20, 20)
        fig.suptitle("Plot of rates of change of joint angles for: \"" + self.ja_file_name +
                     "\" (Row = joint angle, Column = dimension (x, y, or z) of angle): "
                     "X-axis = time(s), Y-axis = angle change per sample", size=13)
        plt.show()


    def write_statistical_features(self, output_name=None, split_file=1):
        """
            :param 'output_name' defaults to the short name of the joint angle file, which writes to an individual
            file for the object (i.e. a single line .csv for the object); specify shared 'output_name' to
            instead append to an existing .csv:
            :return: no return, but creates (or appends, see above) a row to a .csv containing statistical info
            for each of the features and each of their dimensions
        """
        x_angles, y_angles, z_angles = self.x_angles, self.y_angles, self.z_angles
        # Angles has shape: (# of dimensions of features (3 for x,y, and z) x # features (22 here) x # samples (~22K),
        # hence angles[i][j] selects list of sample values for dimension 'i' for feature 'j'
        angles = np.swapaxes(np.asarray([x_angles.T, y_angles.T, z_angles.T]), 0, 1)
        split_size = int(len(angles[0, 0]) / split_file)
        written_names = []
        for f in range(split_file):
            ang = angles[:, :, (split_size * f):(split_size * (f + 1))]
            df = extract_stats_features(f=f, split_file=split_file, file_name=self.ja_file_name,
                                        measurement_names=["jointAngle"], file_part=[ang])
            file_prefix = "DC" if self.is_dc_object else "JA"
            if not output_name:
                output_complete_name = output_dir + file_prefix + "_" + self.ja_file_name + "_stats_features.csv"
                print("Writing", file_prefix, self.ja_file_name, "(", (f+1) , "/", split_file,
                      ") statistial info to", output_complete_name)
            else:
                output_complete_name = output_dir + file_prefix + "_" + output_name + "_stats_features.csv"
                print("Writing", file_prefix, self.ja_file_name, "(", (f+1), "/",
                      split_file, " ) statistial info to", output_complete_name)
            if isfile(output_complete_name):
                with open(output_complete_name, 'a') as file:
                    df.to_csv(file, header=False)
            else:
                with open(output_complete_name, 'w') as file:
                    df.to_csv(file, header=True)
            written_names.append(output_complete_name)
        return written_names


    def write_direct_csv(self, output_name=None):
        """
            :param 'output_name', which, if specified, ensures the file is written to a specified file name rather
            than a name dependent on the source file name (e.g. 'D2')
            :return: no return, but instead directly writes a JointAngleFile object from a .mat file to a .csv
            without extracting any of the statistical features
        """
        df = pd.DataFrame(self.ja_data)
        headers = [("(" + joint_labels[i] + ") : (" + axis_labels[j] + "-axis)")
                   for i in range(len(joint_labels)) for j in range(len(axis_labels))]
        file_prefix = "DC" if self.is_dc_object else "JA"
        if not output_name:
            output_complete_name = output_dir + file_prefix + "_" + self.ja_file_name + ".csv"
            print("Writing", file_prefix, self.ja_file_name, "to", output_complete_name)
        else:
            output_complete_name = output_dir + file_prefix + "_" + output_name + ".csv"
            print("Writing", file_prefix, self.ja_file_name, "to", output_complete_name)
        if isfile(output_complete_name):
            with open(output_complete_name, 'a') as file:
                df.to_csv(file, header=False, index=False)
        else:
            with open(output_complete_name, 'w') as file:
                df.to_csv(file, header=headers, index=False)
        return output_complete_name




def class_selector(ft, fn, fns, is_all, split_files, is_extract_csv=False):
    names = []
    if ft == "ad":
        if is_all:
            for f in fns:
                names += AllDataFile(f).write_statistical_features(output_name="all", split_file=split_files)
        else:
            names.append(AllDataFile(fn).write_statistical_features(split_file=split_files))
    elif ft == "ja":
        if not is_extract_csv:
            if is_all:
                for f in fns:
                    names += JointAngleFile(f).write_statistical_features(output_name="all", split_file=split_files)
            else:
                names += JointAngleFile(fn).write_statistical_features(split_file=split_files)
        else:
            if is_all:
                for f in fns:
                    names += JointAngleFile(f).write_direct_csv(output_name="all")
            else:
                names += JointAngleFile(fn).write_direct_csv()
    else:
        names += DataCubeFile().write_statistical_features()
    return list(dict.fromkeys(np.ravel(names)))



parser = argparse.ArgumentParser()
parser.add_argument("ft", help="Specify type of file file we wish to read from, being one of 'JA' (joint angle), "
                               "'AD' (all data), or 'DC' (data cube).")
parser.add_argument("fn", help="Specify the short file name to load; e.g. for file 'All-HC2-6MinWalk.mat' or "
                               "jointangleHC2-6MinWalk.mat, enter 'HC2'. Specify 'all' for all the files available "
                               "in the default directory of the specified file type. Enter anything for 'dc' file type.")
parser.add_argument("--dis_3d_pos", type=bool, nargs="?", const=True,
                    help="Plots the dynamic positions of an AllDataFile object over time. "
                         "Only works with 'ft' set as an 'AD' file name.")
parser.add_argument("--dis_diff_plot", type=bool, nargs="?", const=True,
                    help="Plots the diff plots of all features and axes of a JointAngleFile object over time."
                         "Only works with 'ft' set as a 'JA' file name.")
parser.add_argument("--dis_3d_angs", type=bool, nargs="?", const=True,
                    help="Plots the 3D positions of all features' joint angles over time. "
                         "Only works with 'ft' set as a 'JA' file name.")
parser.add_argument("--split_files", type=int, help="Splits each of the files into 'n' number of parts to do statistical "
                                          "analysis on, with each retaining the original file's file label ('HC' or 'D').")
parser.add_argument("--check_for_abnormalities", type=float, nargs="?", const=True,
                    help="Checks for abnormalities within the currently-working-on file within it's parts. In other words,"
                         "if '--split_files=5', checks whether any of the 5 file parts is significantly different from "
                         "any of the other parts.")
parser.add_argument('--extract_csv', type=bool, nargs="?", const=True,
                    help="Directly writes a JA file from .mat to .csv format (i.e. without stat extraction).")
args = parser.parse_args()


split_files = args.split_files if args.split_files else 1
if not args.dis_3d_pos and not args.dis_diff_plot and not args.dis_3d_angs:
    names = []
    if args.ft in file_types:
        file_names = []
        if args.ft.lower() == "ad":
            file_names = ad_short_file_names
        elif args.ft.lower() == "ja":
            file_names = ja_short_file_names
        else:
            file_names = None

        if file_names:
            if args.fn in file_names:
                names = class_selector(args.ft, args.fn, None, False, split_files=split_files,
                                       is_extract_csv=args.extract_csv)
            elif args.fn == "all":
                names = class_selector(args.ft, None, file_names, True, split_files=split_files,
                                       is_extract_csv=args.extract_csv)
            else:
                print("Second arg ('fn') must be one of the file names for the '" + args.ft + "' file type, or 'all'")
                sys.exit(1)
        else:
            DataCubeFile().write_statistical_features(split_file=split_files)
    else:
        print("First arg ('ft') must be one of the accepted file types ('ja', 'ad', or 'dc').")
        sys.exit(1)
    if args.check_for_abnormalities:
        check_for_abnormality(names, error_margin=args.check_for_abnormalities)




elif args.dis_3d_pos:
    if args.ft != "ad":
        print("First arg ('ft') must be 'ad' for calling 'display_3d_positions' method.")
        sys.exit(1)
    else:
        if args.fn in ad_short_file_names:
            AllDataFile(args.fn).display_3d_positions()
        else:
            print("Second arg ('fn') must be the short name of an all data file (e.g. 'D2', 'HC5').")
            sys.exit(1)
elif args.dis_diff_plot:
    if args.ft != "ja":
        print("First arg ('ft') must be 'ja' for calling 'display_diffs_plot' method.")
        sys.exit(1)
    else:
        if args.fn in ja_short_file_names:
            JointAngleFile(args.fn).display_diffs_plot()
        else:
            print("Second arg ('fn') must be the short name of a joint angle file (e.g. 'D2').")
            sys.exit(1)
elif args.dis_3d_angs:
    if args.ft != "ja":
        print("First arg ('ft') must be 'ja' for calling 'display_3d_angles' method.")
        sys.exit(1)
    else:
        if args.fn in ja_short_file_names:
            JointAngleFile(args.fn).display_3d_angles()
        else:
            print("Second arg ('fn') must be the short name of a joint angle file (e.g. 'D2').")
            sys.exit(1)
