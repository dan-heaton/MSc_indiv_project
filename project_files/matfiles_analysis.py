import scipy.io as sio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import sys
from os.path import isfile

sampling_rate = 60      #In Hz

#Note: CHANGE THESE to location of '6minwalk-matfiles' directory local to the user
source_dir = "C:\\"
output_dir = "output_files\\"

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

def fft_round(nums, shape):
    """
    :param list of 2d values of which we want to find the 2-dimensional FFT:
    :return result of 2D fft operation with shape given by 'shape' param and values rounded
    """
    efeft = np.fft.fft2(nums, s=shape)
    return [[round(efeft[i][j], 4) for j in range(len(efeft[i]))] for i in range(len(efeft))]



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


    def write_statistical_features(self, measurements_to_extract=None, output_name=None):
        """
            :param features_to_extract to be an optional list of measurement names from the AD file that we wish to
            apply statistical analysis (otherwise, defaults to 'measurement_names'), along with 'output_name' that
            defaults to the short name of the AD file, which writes to an individual file for the object (i.e.
            a single line .csv for the object); specify shared 'output_name' to instead append to an existing .csv:
            :return no return, but writes to an output .csv file the some of the statistical features of the certain
            measurements to an output .csv file
        """
        data = {}
        #Sets the default measurements to extract from the AD file object if none are specified as args
        measurement_names = measurements_to_extract if measurements_to_extract else [
            "position", "sensorFreeAcceleration", "sensorMagneticField", "velocity", "angularVelocity",
            "acceleration", "angularAcceleration"]
        #Disregard first 3 samples (usually types 'identity', 'tpose' or 'tpose-isb')
        #Note that the somewhat-superfluous 's for s' is needed to force 'extract_data' to interpret data as 3D array
        #of shape (# measurements)x21810x(3*num_features) rather than 2D array of shape (# measurements)x21810x(unknown)
        extract_data = [np.asarray([s for s in self.df.loc[3:, feature_name]]) for feature_name in measurement_names]
        # Extracts values of the various measurements in different layout so 'f_d_s_data' now has shape:
        # (# measurements (e.g. 3 for position, accel, and megnet field) x # of features (e.g. 23 for segments)
        # x # of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K))
        f_d_s_data = [[[measurement[:, (i*3)+j] for j in range(3)] for i in range(int(len(measurement[0])/3))]
                      for measurement in extract_data]

        for i, measurement in enumerate(f_d_s_data):        #For each measurement category
            for j in range(len(f_d_s_data[i])):             #For each feature (e.g. segment, joint, or sensor)
                for k in range(len(f_d_s_data[i][j])):      #For each x,y,z dimension
                    data[segment_labels[j] + ": " + axis_labels[k] + "-axis " + measurement_names[i] + " mean"] = \
                        mean_round(measurement[j][k])
                    data[segment_labels[j] + ": " + axis_labels[k] + "-axis " + measurement_names[i] + " variance"] = \
                        variance_round(measurement[j][k])
                    data[segment_labels[j] + ": " + axis_labels[k] + "-axis " + measurement_names[i] +
                        " abs mean sample diff"] = mean_diff_round(measurement[j][k])
                data[segment_labels[j] + ": " + "(x,y,z)" + measurement_names[i] + "2D FFT"] = \
                    fft_round(measurement[j], shape=self.fft_shape)

        # Creates a DataFrame object from a single list version of the dictionary (so it's only 1 row),
        # and creates either a .csv with the default output_name (w/ .csv name from the AD short file name)
        # or with a given name and, if the given name already exists, appends it to the end of existing file
        df = pd.DataFrame([data], columns=data.keys(), index=[self.ad_file_name])
        if not output_name:
            output_complete_name = output_dir + "AD_" + self.ad_file_name + "_stats_features.csv"
            print("Writing AD", self.ad_file_name, "statistial info to", output_complete_name)
            df.to_csv(path_or_buf=output_complete_name, sep=",", mode="w")
        else:
            output_complete_name = output_dir + "AD_" + output_name + "_stats_features.csv"
            print("Writing AD", self.ad_file_name, "statistial info to", output_complete_name)
            if isfile(output_complete_name):
                with open(output_complete_name, 'a') as f:
                    df.to_csv(f, header=False)
            else:
                with open(output_complete_name, 'w') as f:
                    df.to_csv(f, header=True)





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


    def write_statistical_features(self, output_name="All"):
        """
        :param 'output_name', which defaults to 'All', which is the short name of the output .csv stats file that the
        objects will share...specify a different name if desired
        :return no return, but creates a single .csv output file that contains the statistical analysis of each joint
        angle file contained within the data cube via calls to JointAngleFile.write_statistical_features() method
        for each of the joint angle objects extracted at initialization
        """
        for i in range(len(self.ja_objs)):
            self.ja_objs[i].write_statistical_features(output_name=output_name)



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
            ja_data = sio.loadmat(ja_dir + "data_cube_6mw",
                                  matlab_compatible=True)["data_cube"][0][0][2][0][dc_object_index]
        else:
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
        file_prefix = "DC" if self.is_dc_object else "JA"
        if not output_name:
            output_complete_name = output_dir + file_prefix + "_"+ self.ja_file_name + "_stats_features.csv"
            print("Writing", file_prefix, self.ja_file_name, "statistial info to", output_complete_name)
            df.to_csv(path_or_buf=output_complete_name, sep=",", mode="w")
        else:
            output_complete_name = output_dir + file_prefix + "_" + output_name + "_stats_features.csv"
            print("Writing", file_prefix, self.ja_file_name, "statistial info to", output_complete_name)
            if isfile(output_complete_name):
                with open(output_complete_name, 'a') as f:
                    df.to_csv(f, header=False)
            else:
                with open(output_complete_name, 'w') as f:
                    df.to_csv(f, header=True)


for fn in ad_short_file_names:
    AllDataFile(fn).write_statistical_features(output_name="All")
for fn in ja_short_file_names:
    JointAngleFile(fn).write_statistical_features(output_name="All")

DataCubeFile().write_statistical_features()
