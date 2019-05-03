import scipy.io as sio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import sys
from os.path import isfile

sampling_rate = 60      #In Hz

#Note: CHANGE THIS to location of '6minwalk-matfiles' directory local to the user
source_dir = "C:\\"

ja_dir = source_dir + "6minwalk-matfiles\\joint_angles_only_matfiles\\"
ad_dir = source_dir + "6minwalk-matfiles\\all_data_mat_files\\"
joint_angle_filenames = ["D2", "D3", "D4", "HC2", "HC5", "HC6"]
axis_labels = ["X", "Y", "Z"]

joint_labels = ["jL5S1", "jL4L3", "jL1T12", "jT9T8", "jT1C7", "jC1Head", "jRightT4Shoulder", "jRightShoulder",
                "jRightElbow", "jRightWrist", "jLeftT4Shoulder", "jLeftShoulder", "jLeftElbow", "jLeftWrist",
                "jRightHip", "jRightKnee", "jRightAnkle", "jRightBallFoot", "jLeftHip", "jLeftKnee",
                "jLeftAnkle", "jLeftBallFoot"]
segment_labels = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head", "RightShoulder", "RightUpperArm",
                  "RightForeArm", "RightHand", "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
                  "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe", "LeftUpperLeg", "LeftLowerLeg",
                  "LeftFoot", "LeftToe"]
sensor_labels = ["Pelvis", "T8", "Head", "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
                 "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand", "RightUpperLeg", "RightFoot",
                 "LeftUpperLeg", "LegLowerLeg", "LeftFoot"]


"""
Section below covers a few general-purpose mathematical functions used for several file object types 
for statistical analysis
"""

def compute_diffs(nums):
    """
        :param list of values of which we wish to find the diffs between each successive value:
        :return list of diff values of len = len(nums)-1, where each value is difference between value
        in the 'nums' list and the next value in the list:
    """
    return [(nums[i+1]-nums[i]) for i in range(len(nums)-1)]

def mean_round(nums):
    """
        :param list of values of which we wish to find the mean/average:
        :return a float value rounded to 2 decimal places of the mean of 'nums':
    """
    return round(float(np.mean(nums)), 3)

def variance_round(nums):
    """
        :param list of values of which we wish to find the variance:
        :return a float value rounded to 2 decimal places of the variance of 'nums':
    """
    return round(float(np.var(nums)), 3)

def covariance_round(nums):
    """
        :param list of values of 2D array of numbers to find covariance of (shape = # of dimensions of data (e.g. 3 for
        x,y,z data) x # of samples):
        :return covariance matrix of rounded float values of shape = # of dimensions of data x # of dimensions of data:
    """
    return [[round(float(n), 3) for n in num] for num in np.cov(nums.tolist())]

def mean_diff_round(nums):
    """
        :param list of values of which we wish to find the mean diff values:
        :return mean value of the absolute values of the diffs list (see 'compute_diffs'):
    """
    return round(float(np.mean(np.absolute(compute_diffs(nums)))), 3)




class AllDataFile(object):

    def __init__(self, ad_file_name):
        """
            :param string name of the unique identifier of an 'All-' data matlab file (e.g. 'HC3'):
            :return no return, but sets up the DataFrame 'df' attribute from the given matlab file: name
        """
        self.ad_file_name = ad_file_name
        # Loads the data from the given file name and extracts the root 'tree' from the file
        ad_data = sio.loadmat(ad_dir + "All-" + ad_file_name + "-6MinWalk")
        tree = ad_data["tree"]
        # Corresponds to location in matlabfile at: 'tree.subjects.frames.frame'
        frame_data = tree[0][0][6][0][0][10][0][0][3][0]
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
        #(# of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K) x # of features (23 for segments)
        xyz_pos = [[[s for s in sample[i::3]] for sample in positions] for i in range(3)]
        #Gets the max and min of each dimension along all samples and features
        xzy_min_max = [min(np.ravel(xyz_pos[0])), max(np.ravel(xyz_pos[0])),
                       min(np.ravel(xyz_pos[1])), max(np.ravel(xyz_pos[1])),
                       min(np.ravel(xyz_pos[2])), max(np.ravel(xyz_pos[2]))]

        ax = plt.figure().add_subplot(111, projection='3d')
        #Sets the proportions of the plotted graph (so the 'x' dimension is longer than the others, which is approx
        #direction the positions vary most over time)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2, 0.5, 0.5, 1]))
        plt.ion()
        # Contains pairs of feature values which are connected (e.g. left toe to left foot)
        point_connections = [(0, 1), (0, 15), (0, 19), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                             (7, 8), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14), (15, 16),
                             (16, 17), (17, 18), (19, 20), (20, 21), (21, 22)]
        #For each sample to plot, reset the axis limits, plot the relevant 3D points for each feature, plot their
        #scatter point connections (e.g. head to neck, left arm to left shoulder, etc.), updates the frame and
        #time counter, and repeat every (1/sampling_rate) seconds
        for i in range(len(positions)):
            ax.set_xlim(xzy_min_max[0], xzy_min_max[1])
            ax.set_ylim(-4, -2)
            ax.set_zlim(xzy_min_max[4], xzy_min_max[5])
            ax.scatter(xyz_pos[0][i], xyz_pos[1][i], xyz_pos[2][i])
            for pair in point_connections:
                ax.plot([xyz_pos[0][i][pair[0]], xyz_pos[0][i][pair[1]]],
                        [xyz_pos[1][i][pair[0]], xyz_pos[1][i][pair[1]]],
                        [xyz_pos[2][i][pair[0]], xyz_pos[2][i][pair[1]]])
            plt.title("Frame: " + str(i + 1) + ", time = " + str(round(((i + 1) / sampling_rate), 2)) + "s")
            plt.pause(1 / sampling_rate)
            ax.clear()


    def file_info(self):
        """
            :param string name of the unique identifier of an 'All-' data matlab file (e.g. 'HC3'):
            :return no return, but sets up the DataFrame 'df' attribute from the given matlab file: name
        """
        print("\nAd " + self.ad_file_name + " keys:", ", ".join([("'" + key + "'") for key in self.df]), "\n")
        for k in list(self.df.keys())[:3]:
            print(k, ":", (self.df[k]))





class DataCubeFile(object):

    def __init__(self, dis_data_cube=False):
        """
            :param dis_data_cube is specified to True if wish to display basic Data Cube info to the screen:
            :return no return, but sets up the Data Cube attribute
        """
        ja_dir = "6minwalk-matfiles\\joint_angles_only_matfiles\\"
        self.dc = sio.loadmat(ja_dir + "data_cube_6mw")

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





class JointAngleFile(object):

    def __init__(self, ja_file_name):
        """
            :param string name of the unique identifier of an 'jointangle-' data matlab file (e.g. 'D2'):
            :return no return, but extracts the data from the specified matlab file and extracts the features
            from each dimension
        """
        self.ja_file_name = ja_file_name
        ja_p = "jointangle"
        ja_s = "-6MinWalk"
        #'Try-except' clause included to handle some slight naming inconsistencies with the JA filename syntax
        try:
            ja_data = sio.loadmat(ja_dir + ja_p + ja_file_name + ja_s)['jointangle']
        except FileNotFoundError:
            ja_data = sio.loadmat(ja_dir + ja_p + ja_file_name + ja_s + "-TruncLen5Min20Sec")['jointangle']

        #x/y/z_angles arrays have shape (# of samples in JA data file x # of features (i.e. # of features, 22))
        self.x_angles = np.asarray([[s for s in sample[0::3]] for sample in ja_data])
        self.y_angles = np.asarray([[s for s in sample[1::3]] for sample in ja_data])
        self.z_angles = np.asarray([[s for s in sample[2::3]] for sample in ja_data])
        #xyz_angles array has shape (# of samples in JA data file x # of features (i.e. # of features, 22)
        # x # of dimensions of data (i.e. 3 for x,y,z))
        self.xyz_angles = np.asarray([[[x, y, z] for x, y, z in
                                       zip([s for s in sample[0::3]], [s for s in sample[1::3]],
                                           [s for s in sample[2::3]])] for sample in ja_data])


    def display_3d_angles(self):
        """
            :param no params
            :return no return, but displays the angles of each feature in 3D as it changes over time (i.e. with sample num)
        """
        ja_3d_data = self.xyz_angles
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        for sample in ja_3d_data:
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
            plt.pause((1/sampling_rate))
            ax.clear()


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


    def write_statistical_features(self, output_name=None):
        """
            :param 'output_name' defaults to the short name of the joint angle file, which writes to an individual
            file for the object (i.e. a single line .csv for the object); specify shared 'output_name' to
            instead append to an existing .csv:
            :return: no return, but creates (or appends, see above) a row to a .csv containing statistical info
            for each of the features and each of their dimensions
        """
        data = OrderedDict()
        x_angles, y_angles, z_angles = self.x_angles, self.y_angles, self.z_angles

        # Angles has shape: (# of dimensions of features (3 for x,y, and z) x # features (22 here) x # samples (~22K),
        # hence angles[i][j] selects list of sample values for dimension 'i' for feature 'j'
        angles = np.asarray([x_angles.T, y_angles.T, z_angles.T])
        #Adds mean, variance, covariance, and abs mean sample diff info to the 'data' dictionary
        #for each feature's dimension
        for i in range(len(angles[0])):     #For each feature
            for j in range(len(angles)):    #For each dimension
                data[joint_labels[i] + ": " + axis_labels[j] + "-axis mean"] = mean_round(angles[j][i])
                data[joint_labels[i] + ": " + axis_labels[j] + "-axis variance"] = variance_round(angles[j][i])
                data[joint_labels[i] + ": " + axis_labels[j] + "-axis abs mean sample diff"] = \
                    mean_diff_round(angles[j][i])
            # angles[:, i] selects list of sample values for all 3 dimensions for feature 'i'
            data[joint_labels[i] + ": (x,y,z) covariance"] = covariance_round(angles[:, i])
        #Computes mean for each dimension (x, y, or z) but over every single feature and adds to the 'data' dictionary
        for i in range(len(axis_labels)):
            data["(Over all features): " + axis_labels[i] + "-axis mean"] = \
                mean_round([data[k] for k in data.keys() if axis_labels[i] + "-axis mean" in k])
            data["(Over all features): " + axis_labels[i] + "-axis variance"] = \
                mean_round([data[k] for k in data.keys() if axis_labels[i] + "-axis variance" in k])
            data["(Over all features): " + axis_labels[i] + "-axis abs mean sample diff"] = \
                mean_round([data[k] for k in data.keys() if axis_labels[i] + "-axis abs mean" in k])

        #Creates a DataFrame object from a single list version of the dictionary (so it's only 1 row),
        #and creates either a .csv with the default output_name (w/ .csv name from the JA short file name)
        #or with a given name and, if the given name already exists, appends it to the end of existing file
        df = pd.DataFrame([data], columns=data.keys(), index=[self.ja_file_name])
        if not output_name:
            df.to_csv(path_or_buf="output_files\\JA_" + self.ja_file_name + "_stats_features.csv", sep=",", mode="w")
        else:
            file_loc = "output_files\\JA_" + output_name + "_stats_features.csv"
            if isfile(file_loc):
                with open(file_loc, 'a') as f:
                    df.to_csv(f, header=False)
            else:
                with open(file_loc, 'w') as f:
                    df.to_csv(f, header=True)




#ad_d2 = AllDataFile("D2")
#print(ad_d2.df)
#ad_d2.display_3d_positions()

#ja_d2 = JointAngleFile("D3")
#ja_d2.write_statistical_features("D2")
#ja_d2.display_3d_angles()

#data = ad_extract_frame("D2")
#ad_display_3d_positions(data)
