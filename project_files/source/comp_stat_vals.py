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
import os
from os.path import isfile
import argparse
from settings import sampling_rate, local_dir, sub_dirs, source_dir, short_file_types, axis_labels, segment_labels, \
    joint_labels, sensor_labels, measure_to_len_map, seg_join_sens_map


#Reassigns the name of 'source_dir' to 'output_dir' in this script, as what is used as the source for several
#other scripts is what is used as the output directory in this script
output_dir = source_dir
file_types = short_file_types


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
    """
        :param list of values of which we want to find the 1-dimensional FFT for 3 values
        :return largest FFT value from list of 3 values rounded to 4 decimal place
    """
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
    return [[round(float(n), 8) for n in num] for num in np.cov(nums.tolist())]

def fft_2d_round(nums):
    """
        :param list of 2D values of which we want to find the 2-dimensional FFT for 3 values:
        :return list of 3 values returned from the 2D FFT operation, in descending order and each rounded
        to 4 decimal places
    """
    fft_2d = np.fft.fft2(nums, s=(1, 3))
    fft_2d.sort()
    fft_2d = np.flip(fft_2d)
    return [round(np.real(fft_2d[0][i]), 4) for i in range(len(fft_2d[0]))]

def mean_sum_vals(nums):
    """
        :param list of values of 2D array of numbers of which we wish to find the mean of the sums of each dimension
        :return mean of the sum of each dimension of the nums 2D array (i.e. sum along each x, y, and z of 2D feature
        values), rounded to 4 decimal places
    """
    return round(float(np.mean([np.sum(nums[i]) for i in range(len(nums))])), 4)

def mean_sum_abs_vals(nums):
    """
        :param list of values of 2D array of numbers of which we wish to find the mean of the sums of each dimension
        :return same as 'mean_sum_vals', with difference that each value in 2D input array is 'absoluted' first
    """
    return round(float(np.mean([np.sum(np.abs(nums[i])) for i in range(len(nums))])), 4)

def cov_eigenvals(nums, i):
    """
        :param list of values of 2D array of numbers of which we wish to find the eigenvals of covariance matrix, and an
        index to indicate which of the eigenvals we wish to return
        :return one of the two largest (rounded to 4 decimal places) eigenvalues of the covariance matrix of the
        2D array of numbers (i.e. the top 2 of 3 eigenvals), depending on the index 'i'
    """
    vals = la.eig(covariance_round(nums))[0].tolist()
    vals = list(vals)
    vals.sort(reverse=True)
    top_vals = vals[:2]
    return round(top_vals[i], 4)



def prop_outside_mean_zone(nums, percen=0.1):
    """
        :param array of 2D numbers of which we wish to find the proportion of the samples in the array that are outside
        'percen' boundaries around the mean zone (e.g. if mean zone is 5 for one dim, the sample would have to have
        a value not between 4.5 and 5.5 for that dimension's value to be considered 'outside'; the same just also
        be true for its other 2 dimensions)
        :return the proportion of the samples within 'nums' that are outside the mean zone (i.e. ALL 3 of its
        lie beyond their respective mean zones
    """
    #Mean values of the array of nums for each of its x, y, and z dimensions; used to calculate if a sample is
    #outside the mean zone
    x_mean, y_mean, z_mean = np.mean(nums[0]), np.mean(nums[1]), np.mean(nums[2])
    nums_outside = 0
    nums = np.swapaxes(nums, 0, 1)
    for num in nums:
        #Each sample must have all 3 of its dimensions within the 'mean zone' to have 1 added to
        if not x_mean*(1-percen) < num[0] < x_mean*(1+percen):
            if not y_mean*(1-percen) < num[1] < y_mean*(1+percen):
                if not z_mean*(1-percen) < num[2] < z_mean*(1+percen):
                    nums_outside += 1
    return round(nums_outside/len(nums), 4)



def extract_stats_features(f_index, file_part, split_file, file_name, measurement_names, is_recursive_call=False):
    """

    :param 'f_index' for a number the file_part represents of the complete file, 'file_part' being the data itself that
    we will be extracting the statistical features of, 'split_file' being the total number of parts within the complete
    file that 'file_part' comes from, 'file_name' being the name of the file that 'file_part' comes from,
    'measurement_names' being a list of measurements (e.g. angularAcceleration, position, jointAngles, etc.) we wish to
    extract the statistical features from, and 'is_recursive_call' used to handle the recursive case
    :return: dictionary of data containing the complete statistical features extracted for a single file part that
    corresponds to a single line written to an output .csv
    """
    data = {}
    # Adds a 'y' label for a given file
    data['file_type'] = "HC" if "HC" in file_name else "D"
    #For each measurement category
    for i in range(len(file_part)):
        seg_join_sens_labels = seg_join_sens_map[measure_to_len_map[measurement_names[i]]]
        num_features = 1 if is_recursive_call else measure_to_len_map[measurement_names[i]]
        #For each feature (e.g. segment, joint, or sensor)
        for j in range(num_features):
            #For each x,y,z dimension
            for k in range(len(file_part[i][j])):
                #Gives each statistical value calculated for a specific measumrent/feature/axis combination a
                #label based on these values that are shared amongst all statistical extracted values
                if is_recursive_call:
                    stat_name = "(" + measurement_names[i] + ") : (Over all features) : (" + axis_labels[k] + "-axis)"
                else:
                    stat_name = "(" + measurement_names[i] + ") : (" + seg_join_sens_labels[j] + \
                                ") : (" + axis_labels[k] + "-axis)"
                data[stat_name + " : (mean)"] = mean_round(file_part[i][j][k])
                data[stat_name + " : (variance)"] = variance_round(file_part[i][j][k])
                data[stat_name + " : (abs mean sample diff)"] = mean_diff_round(file_part[i][j][k])
                data[stat_name + " : (FFT largest val)"] = fft_1d_round(file_part[i][j][k])
            #For column labels for statistical values computed over all 3 dimensions, gives a shared label part based
            #just on the measurement name and feature being computed in question that's shared over all
            #statistical extracted values
            if is_recursive_call:
                xyz_stat_name = "(" + measurement_names[i] + ") : (Over all features) : ((x,y,z)-axis) : "
            else:
                xyz_stat_name = "(" + measurement_names[i] + ") : ( " + seg_join_sens_labels[j] + ") : ((x,y,z)-axis) : "
            data[xyz_stat_name + "(mean sum vals)"] = mean_sum_vals(file_part[i][j])
            data[xyz_stat_name + "(mean sum abs vals)"] = mean_sum_abs_vals(file_part[i][j])
            data[xyz_stat_name + "(first eigen cov)"] = cov_eigenvals(file_part[i][j], 0)
            data[xyz_stat_name + "(second eigen cov)"] = cov_eigenvals(file_part[i][j], 1)
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
    # '# measurements' x 1 x '# axes' x '(# samples x # features)' to form 'new_file_part'
    new_file_part = np.reshape(file_part, (len(file_part), 1, len(file_part[0][0]), -1))
    data_over_all = extract_stats_features(f_index=f_index, file_part=new_file_part, split_file=split_file,
                                           file_name=file_name, measurement_names=measurement_names, is_recursive_call=True)
    #Adds the statistical values that are computed in the recursive case (i.e. computed inter-features rather than
    #intra-features) to the original dictionary
    data.update(data_over_all)

    # Creates and returns a DataFrame object from a single list version of the dictionary (so it's only 1 row),
    # and creates either a .csv with the default output_name (w/ .csv name from the AD short file name)
    # or with a given name and, if the given name already exists, appends it to the end of existing file
    return pd.DataFrame([data], columns=data.keys(), index=[file_name + "_(" +
                                                            str(f_index + 1) + "/" + str(split_file) + ")"])



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

    def __init__(self, ad_file_name, sub_dir):
        """
            :param 'ad_file_name' being the string name of the complete name of the source file (i.e. full file
            path name), with 'short_fn' being the string name of the unique identifier of an 'All-' data matlab file
            (e.g. 'HC3'), and 'sub_dir' being the name of the sub-directory within 'output_dir' to place the
            extracted statistical values when 'write_statistical_features' is called
            :return no return, but sets up the DataFrame 'df' attribute from the given matlab file
        """
        self.ad_file_name = ad_file_name
        split_name = ad_file_name.split("\\")[-1].split("-")
        splits = [[s for s in split_name if "HC" in s], [s for s in split_name if "D" in s or "d" in s]]
        self.ad_short_file_name = splits[1][0] if not splits[0] else splits[0][0]
        self.ad_sub_dir = sub_dir
        # Loads the data from the given file name and extracts the root 'tree' from the file
        ad_data = sio.loadmat(ad_file_name)
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
        try:
            frame_data = [[elem[0] if len(elem[0]) != 1 else elem[0][0] for elem in row] for row in frame_data]
        except IndexError:
            #Accounts for missing 'contact' values in certain rows of some '.mat' files by ignoring the 'contact' column
            new_frame_data = []
            for m, row in enumerate(frame_data):
                #Ignore rows that don't have 'normal' as their 'type' cell
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
        df = pd.DataFrame(frame_data, columns=col_names)
        self.df = df


    def display_3d_positions(self):
        """
            :param no params, but relies on the 'df' attribute that is set up at the 'AllDataFile' object's instantiation
            :return no return, but plots position values for the given file object on a 3D dynamic plot,
            with the necessary components connected, and outputs a dynamic accumulated time in seconds to the console:
        """
        #Disregard first 3 samples (usually types 'identity', 'tpose' or 'tpose-isb')
        positions = self.df.loc[:, "position"].values[3:]
        #Extracts position values for each dimension so 'xyz_pos' now has shape:
        # # of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K) x # of features (23 for segments)
        xyz_pos = [[[s for s in sample[i::3]] for sample in positions] for i in range(3)]
        #Xyz_pos_t now has dimensions:
        # # of features (23 for segments) x # of samples (~22K) x # of dimensions (e.g. 3 for x,y,z position values)
        xyz_pos_t = np.swapaxes(xyz_pos, 0, 2)
        #List of tuples that define how much segment labels are connected on the human body (e.g. segments 0 and 1 are
        #connected as are 0 and 15) so as to dynamically drawn lines between them on the 3D plot
        stick_defines = [(0, 1), (0, 15), (0, 19), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                         (7, 8), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14), (15, 16),
                         (16, 17), (17, 18), (19, 20), (20, 21), (21, 22)]
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        #Calculates the maximums and means alone each x, y, and z dimension to use to set the 3D boundaries
        maxs = [max([max(xyz_pos[i][j]) for j in range(len(xyz_pos[i]))]) for i in range(len(xyz_pos))]
        mins = [min([min(xyz_pos[i][j]) for j in range(len(xyz_pos[i]))]) for i in range(len(xyz_pos))]
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        xy_ratio = (maxs[1]-mins[1])/(maxs[0]-mins[0])
        xz_ratio = (maxs[2]-mins[2])/(maxs[0]-mins[0])
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, xy_ratio, xz_ratio, 0.25]))

        colors = plt.cm.jet(np.linspace(0, 1, len(xyz_pos[0][0])))
        lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
        pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

        stick_lines = [ax.plot([], [], [], 'k-')[0] for _ in stick_defines]

        #Defines the 'init' function object to setup the points and lines defining a person in 3d space
        def init():
            for line, pt in zip(lines, pts):
                line.set_data([], [])
                line.set_3d_properties([])
                pt.set_data([], [])
                pt.set_3d_properties([])
            return lines + pts + stick_lines

        #Function object that updates the plot based on the next sample of 3D coordinates for each feature (i.e.
        #each 'point' on the walking figure in the plot)
        def animate(i):
            if i >= len(xyz_pos[0]):
                exit()
            if (i*5)%sampling_rate == 0:
                print("Plotting time: " + str(int((i*5)/sampling_rate)) + "s")
            i = (5 * i) % xyz_pos_t.shape[1]
            for line, pt, xi in zip(lines, pts, xyz_pos_t):
                x, y, z = xi[:i].T
                pt.set_data(x[-1:], y[-1:])
                pt.set_3d_properties(z[-1:])
            for stick_line, (sp, ep) in zip(stick_lines, stick_defines):
                stick_line._verts3d = xyz_pos_t[[sp, ep], i, :].T.tolist()
            fig.canvas.draw()
            if (i+5) >= int(len(xyz_pos[0])):
                plt.close()
                print("Animation ended...")
            return lines + pts + stick_lines

        #Plot the figure in 3D space, with an update interval (defined in miliseconds) to be in real time
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


    def write_statistical_features(self, measurements_to_extract=None, output_name=None, split_file=1, split_size=None):
        """
            :param 'measurements_to_extract' to be an optional list of measurement names from the AD file that we wish to
            apply statistical analysis (otherwise, defaults to 'measurement_names'), along with 'output_name' that
            defaults to the short name of the AD file, which writes to an individual file for the object (i.e.
            a single line .csv for the object); specify shared 'output_name' to instead append to an existing .csv;
            'split_file' and 'split_size' optionally given as 2 different ways to break up a file into parts
            :return no return, but writes to an output .csv file the statistical values of the certain
            measurements to an output .csv file
        """
        #Sets the default measurements to extract from the AD file object if none are specified as args
        measurement_names = measurements_to_extract if measurements_to_extract else [
            "position", "sensorFreeAcceleration", "sensorMagneticField", "velocity", "angularVelocity",
            "acceleration", "angularAcceleration"]

        # Extracts values of the various measurements in different layout so 'f_d_s_data' now has shape:
        # (# measurements (e.g. 3 for position, accel, and megnet field) x # of features (e.g. 23 for segments)
        # x # of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K))
        if args.single_act:
            extract_data = self.df.loc[:, measurement_names].values
            f_d_s_data = np.zeros((len(measurement_names),
                                   max(len(segment_labels), len(joint_labels), len(sensor_labels)), 3,
                                   len(self.df.loc[:])))
            for i in range(len(measurement_names)):
                for j in range(int(len(self.df.loc[:, measurement_names[i]].values[0])/3)):
                    for k in range(len(axis_labels)):
                        for m in range(len(self.df.loc[:])):
                            f_d_s_data[i, j, k, m] = extract_data[m, i][(j*3)+k]
        else:
            extract_data = self.df.loc[3:, measurement_names].values
            f_d_s_data = np.zeros((len(measurement_names),
                                   max(len(segment_labels), len(joint_labels), len(sensor_labels)), 3,
                                   len(self.df.loc[:]) - 3))
            for i in range(len(measurement_names)):
                for j in range(int(len(self.df.loc[3:, measurement_names[i]].values[0]) / 3)):
                    for k in range(len(axis_labels)):
                        for m in range(len(self.df.loc[3:])):
                            f_d_s_data[i, j, k, m] = extract_data[m, i][(j * 3) + k]

        #If 'split_size' is given as command line argument and 'split_file' isn't, set 'split_s' to the given number
        #multiplied by sampling rate (i.e. if given num is 5 and 'sampling_rate' is 60, 'split_s' is 300, i.e. the
        #number of rows in each 'split_file') and 'split_file' is set to the number of these sizes that fit in the
        #original file (i.e. how many parts we can get out of a file given a 'split_s')
        if split_file == 1 and split_size:
            split_s = int(sampling_rate * split_size)
            split_file = int(len(f_d_s_data[0, 0, 0])/split_s)
        else:
            split_s = int(len(f_d_s_data[0, 0, 0])/split_file)

        written_names = []
        for i, f in enumerate(range(split_file)):
            fds = f_d_s_data[:, :, :, (split_s*f):(split_s*(f+1))]
            # Creates a dictionary of extracted statistical values for a given file part
            df = extract_stats_features(f_index=f, file_part=fds, split_file=split_file,
                                        file_name=self.ad_short_file_name, measurement_names=measurement_names)

            #Creates the relevant sub-directories within 'output_files' to store the created .csv
            ad_output_dir = output_dir + self.ad_sub_dir + "\\"
            if not os.path.exists(ad_output_dir):
                os.mkdir(ad_output_dir)
            ad_output_dir += "AD\\"
            if args.single_act:
                ad_output_dir += "act_files\\"
            if not os.path.exists(ad_output_dir):
                os.mkdir(ad_output_dir)

            #Sets output name of file (and related print statement) to whether or not the user is storing it in
            #an 'all' .csv or a .csv determined by the name of the writing file
            af = "_" + self.ad_file_name.split(".")[0].split("_")[-1] if args.single_act else ""
            if not output_name:
                output_complete_name = ad_output_dir + "AD_" + self.ad_short_file_name + af + "_stats_features.csv"
                print("Writing AD", self.ad_file_name, "(", (f+1), "/", split_file, ") statistial info to",
                      output_complete_name)
            else:
                output_complete_name = ad_output_dir + "AD_" + output_name + af + "_stats_features.csv"
                print("Writing AD", self.ad_file_name, "(", (f + 1), "/", split_file, ") statistial info to",
                      output_complete_name)
            #Writes just the data to file if the file already exists, or both the data and headers if the file
            #doesn't exist yet
            if isfile(output_complete_name):
                with open(output_complete_name, 'a', newline='') as file:
                    df.to_csv(file, header=False)
            else:
                with open(output_complete_name, 'w', newline='') as file:
                    df.to_csv(file, header=True)
            written_names.append(output_complete_name)
        #Returns the complete names of the file(s) that have been written or appended to a .csv
        return written_names





class DataCubeFile(object):

    def __init__(self, sub_dir, dis_data_cube=False):
        """
            :param 'dis_data_cube' is specified to True if wish to display basic Data Cube info to the screen
            :return no return, but reads in the datacube object from .mat file, the datacube table from the .csv,
            extracts the names of the joint angle files in the datacube file, and instantiates JointAngleFile objects
            for each file contained in the datacube .mat file

            IMPORTANT NOTE: as there's no conceivable way currently to read a matlab table into Python (unlike structs),
            it's required that the matlab table is exported to a .csv before creating DataCubeFileObject. Hence, with
            the data_cube .mat file open in matlab, run writetable(excel_table, "data_cube_table.csv") in matlab for the
            following to work
        """

        self.dc_sub_dir = sub_dir

        try:
            self.dc_table = pd.read_csv(local_dir + "data_cube_table.csv")
            self.dc_short_file_names = self.dc_table.values[:, 1]
            self.dc_file_names = self.dc_table.values[:, 2]
        except FileNotFoundError:
            print("Couldn't find the 'data_cube_table.csv file. Make sure to run the "
                  "'writetable(excel_table, \"data_cube_table.csv\") in matlab with data_cube.mat file open")
            sys.exit()
        self.dc = sio.loadmat(local_dir + "data_cube_6mw", matlab_compatible=True)["data_cube"][0][0][2][0]

        self.ja_objs = []
        for i in range(len(self.dc_short_file_names)):
            print("Extracting data from data cube file (" + str(i+1) + "/" + str(len(self.dc_short_file_names)) +
                  "): " + self.dc_short_file_names[i] + "...")
            self.ja_objs.append(JointAngleFile(ja_file_name=self.dc_file_names[i], sub_dir=self.dc_sub_dir,
                                               dc_object_index=i, dc_short_file_name=self.dc_short_file_names[i]))

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


    def write_statistical_features(self, output_name="all", split_file=1, split_size=None):
        """
        :param 'output_name', which defaults to 'All', which is the short name of the output .csv stats file that the
        objects will share...specify a different name if desired; 'split_file' and 'split_size' optionally given as 2
        different ways to break up a file into parts
        :return no return, but creates a single .csv output file that contains the statistical analysis of each joint
        angle file contained within the data cube via calls to 'JointAngleFile.write_statistical_features()' method
        for each of the joint angle objects extracted at initialization
        """
        written_names = []
        for i in range(len(self.ja_objs)):
            written_names += self.ja_objs[i].write_statistical_features(output_name=output_name, split_file=split_file,
                                                                        split_size=split_size)
        return written_names


    def write_direct_csv(self, output_name="all"):
        """
        :param no params
        :return for each JointAngleFile object within the datacube, call their 'write_direct_csv' method with their
        'output_name' set to their respective short name
        """
        written_names = []
        for i in range(len(self.ja_objs)):
            written_names += self.ja_objs[i].write_direct_csv()
        return written_names


class JointAngleFile(object):

    def __init__(self, ja_file_name, sub_dir, dc_object_index=-1, dc_short_file_name=None):
        """
            :param 'ja_file_name' being the string name of the complete name of the source file (i.e. full file
            path name), with 'short_fn' being the string name of the unique identifier of an 'joint angle' data
            matlab file (e.g. 'D2'), and 'sub_dir' being the name of the sub-directory within 'output_dir' to place
            the extracted statistical values when 'write_statistical_features' is called, and 'dc_object' which is
            set to non-zero if object is created as part of a DataCubeFile:
            :return no return, but sets up the DataFrame 'df' attribute from the given matlab file
        """
        self.ja_file_name = ja_file_name
        split_name = ja_file_name.split("\\")[-1].split("-")
        splits = [[s for s in split_name if "HC" in s], [s.upper() for s in split_name if "D" in s.upper()]]
        if not dc_short_file_name:
            self.ja_short_file_name = splits[1][0][splits[1][0].index("D"):] \
                if not splits[0] else splits[0][0][splits[0][0].index("HC"):]
        else:
            self.ja_short_file_name = dc_short_file_name
        self.ja_sub_dir = sub_dir
        self.is_dc_object = True if dc_object_index != -1 else False
        #Loads the JA data from the datacube file instead of the normal JA files if passed from DataCubeFile class
        if dc_object_index != -1:
            self.ja_data = sio.loadmat(local_dir + "data_cube_6mw",
                                  matlab_compatible=True)["data_cube"][0][0][2][0][dc_object_index]
        else:
            #'Try-except' clause included to handle some slight naming inconsistencies with the JA filename syntax
            try:
                self.ja_data = sio.loadmat(ja_file_name)['jointangle']
            except FileNotFoundError:
                self.ja_data = sio.loadmat(ja_file_name + "-TruncLen5Min20Sec")['jointangle']

        print("Extracting data from JA file " + self.ja_file_name + "....")
        #x/y/z_angles arrays have shape (# of samples in JA data file x # of features (i.e. # of features, 22))
        self.x_angles = np.asarray([[s for s in sample[0::3]] for sample in self.ja_data])
        self.y_angles = np.asarray([[s for s in sample[1::3]] for sample in self.ja_data])
        self.z_angles = np.asarray([[s for s in sample[2::3]] for sample in self.ja_data])
        #xyz_angles array has shape: '# of samples in JA data file' x '# of features (i.e. # of features, 22)'
        # x '# of dimensions of data (i.e. 3 for x,y,z)'
        self.xyz_angles = np.asarray([[[x, y, z] for x, y, z in
                                       zip([s for s in sample[0::3]], [s for s in sample[1::3]],
                                           [s for s in sample[2::3]])] for sample in self.ja_data])


    def display_3d_angles(self):
        """
            :param no params
            :return no return, but displays the angles of each feature in 3D as it changes over
            time (i.e. with sample num)
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
            :return no return, but displays '# of features (e.g. 22 features)' x '# of dimensions (e.g. 3 for x,y,z)'
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


    def write_statistical_features(self, output_name=None, split_file=1, split_size=None):
        """
            :param 'output_name' defaults to the short name of the joint angle file, which writes to an individual
            file for the object (i.e. a single line .csv for the object); specify shared 'output_name' to
            instead append to an existing .csv; 'split_file' and 'split_size' optionally given as 2 different ways
            to break up a file into parts
            :return: no return, but writes to an output .csv file the statistical values of the joint angle
            measurement to an output .csv file
        """
        x_angles, y_angles, z_angles = self.x_angles, self.y_angles, self.z_angles
        #Angles has shape: (# of dimensions of features (3 for x,y, and z) x # features (22 here) x # samples (~22K),
        #hence angles[i][j] selects list of sample values for dimension 'i' for feature 'j'
        angles = np.swapaxes(np.asarray([x_angles.T, y_angles.T, z_angles.T]), 0, 1)

        #If 'split_size' is given as command line argument and 'split_file' isn't, set 'split_s' to the given number
        #multiplied by sampling rate (i.e. if given num is 5 and 'sampling_rate' is 60, 'split_s' is 300, i.e. the
        #number of rows in each 'split_file') and 'split_file' is set to the number of these sizes that fit in the
        #original file (i.e. how many parts we can get out of a file given a 'split_s')
        if split_file == 1 and split_size:
            split_s = int(sampling_rate * split_size)
            split_file = int(len(angles[0, 0])/split_s)
        else:
            split_s = int(len(angles[0, 0])/split_file)

        written_names = []
        for i, f in enumerate(range(split_file)):
            ang = angles[:, :, (split_s * f):(split_s * (f + 1))]
            #Creates a dictionary of extracted statistical values for a given file part
            df = extract_stats_features(f_index=f, split_file=split_file, file_name=self.ja_short_file_name,
                                        measurement_names=["jointAngle"], file_part=[ang])
            file_prefix = "DC" if self.is_dc_object else "JA"

            #Creates the relevant sub-directories within 'output_files' to store the created .csv
            dc_ja_output_dir = output_dir + self.ja_sub_dir + "\\"
            if not os.path.exists(dc_ja_output_dir):
                os.mkdir(dc_ja_output_dir)
            dc_ja_output_dir += file_prefix + "\\"
            if not os.path.exists(dc_ja_output_dir):
                os.mkdir(dc_ja_output_dir)

            #Sets output name of file (and related print statement) to whether or not the user is storing it in
            #an 'all' .csv or a .csv determined by the name of the writing file
            if not output_name:
                output_complete_name = dc_ja_output_dir + file_prefix + "_" + self.ja_short_file_name + "_stats_features.csv"
                print("Writing", file_prefix, self.ja_file_name, "(", (f+1) , "/", split_file,
                      ") statistial info to", output_complete_name)
            else:
                output_complete_name = dc_ja_output_dir + file_prefix + "_" + output_name + "_stats_features.csv"
                print("Writing", file_prefix, self.ja_file_name, "(", (f+1), "/",
                      split_file, " ) statistial info to", output_complete_name)
            #Writes just the data to file if the file already exists, or both the data and headers if the file
            #doesn't exist yet
            if isfile(output_complete_name):
                with open(output_complete_name, 'a', newline='') as file:
                    df.to_csv(file, header=False, line_terminator="")
            else:
                with open(output_complete_name, 'w', newline='') as file:
                    df.to_csv(file, header=True, line_terminator="")
            written_names.append(output_complete_name)
        #Returns the complete names of the file(s) that have been written or appended to a .csv
        return written_names


    def write_direct_csv(self, output_name=None):
        """
            :param 'output_name', which, if specified, ensures the file is written to a specified file name rather
            than a name dependent on the source file name (e.g. 'D2')
            :return: no return, but instead directly writes a JointAngleFile object from a .mat file to a .csv
            without extracting any of the statistical features
        """
        df = pd.DataFrame(self.ja_data, index=[self.ja_short_file_name for i in range(len(self.ja_data))])
        headers = [("(" + joint_labels[i] + ") : (" + axis_labels[j] + "-axis)")
                   for i in range(len(joint_labels)) for j in range(len(axis_labels))]
        file_prefix = "DC" if self.is_dc_object else "JA"

        # Creates the relevant sub-directories within 'output_files' to store the created .csv
        dc_ja_output_dir = output_dir + "direct_csv\\"
        if not os.path.exists(dc_ja_output_dir):
            os.mkdir(dc_ja_output_dir)
        dc_ja_output_dir += file_prefix + "\\"
        if not os.path.exists(dc_ja_output_dir):
            os.mkdir(dc_ja_output_dir)

        if not output_name:
            output_complete_name = dc_ja_output_dir + file_prefix + "_" + self.ja_short_file_name + ".csv"
            print("Writing", file_prefix, self.ja_file_name, "to", output_complete_name)
        else:
            output_complete_name = dc_ja_output_dir + file_prefix + "_" + output_name + ".csv"
            print("Writing", file_prefix, self.ja_file_name, "to", output_complete_name)
        if isfile(output_complete_name):
            with open(output_complete_name, 'a', newline='') as file:
                df.to_csv(file, header=False)
        else:
            with open(output_complete_name, 'w', newline='') as file:
                df.to_csv(file, header=headers)
        return output_complete_name




def class_selector(ft, fn, fns, sub_dir, is_all, split_files, is_extract_csv=False, split_size=None):
    """
        :param 'ft' is the type of data file (i.e. one of 'AD', 'JA', or 'DC'), 'fn' is the full name of the data file
        (i.e. the full-directory path of the file) if 'class_selector' is dealing with a single file, 'fns' is the
        full file names if 'class_selector' is dealing with multiple files(i.e. the 'all'
        command line argument is given for 'fn'), 'sub_dir' is the sub-directory to write the file(s)
        in question in their extracted stats .csv format, 'is_all' is set to true if 'all is given as 'fn' command line
        argument (else false), 'split_files' is set at the value given by '--split_files' command line argument (else 1),
        'is_extract_csv' is true if calling 'write_direct_csv' on JointAngleFile object (else false), and 'split_size'
        is the value set by the '--split_size' command line argument (else None)
        :return list of names that have already been written to output .csv; however, the more important operation is
        the creation of various file objects and the calling on their respective 'write_statistical_features' methods
        with arguments governed by the passed in arguments to 'class_selector'
    """
    names = []
    if ft == "AD":
        if is_all:
            for f in fns:
                names += AllDataFile(f, sub_dir).write_statistical_features(split_file=split_files, split_size=split_size)
        else:
            names.append(AllDataFile(fn, sub_dir).write_statistical_features(
                split_file=split_files, split_size=split_size))
    elif ft == "JA":
        if not is_extract_csv:
            if is_all:
                fns = [fn for fn in fns if "jointangle" in fn]
                for f in fns:
                    names += JointAngleFile(f, sub_dir).write_statistical_features(
                        output_name="all", split_file=split_files, split_size=split_size)
            else:
                names += JointAngleFile(fn, sub_dir).write_statistical_features(
                    split_file=split_files, split_size=split_size)
        else:
            if is_all:
                fns = [fn for fn in fns if "jointangle" in fn]
                for f in fns:
                    names += JointAngleFile(f, sub_dir).write_direct_csv()
            else:
                names += JointAngleFile(fn, sub_dir).write_direct_csv()
    else:
        if not is_extract_csv:
            names += DataCubeFile(sub_dir).write_statistical_features(split_file=split_files, split_size=split_size)
        else:
            names += DataCubeFile(sub_dir).write_direct_csv()
    return list(dict.fromkeys(np.ravel(names)))



def del_files(names):
    for name in names:
        try:
            os.remove(name)
            print("'" + str(name) + "' successfully deleted")
        except FileNotFoundError:
            print("'" + str(name) + "' doesn't exist, unable to delete...")



"""Section below encompases all the command line arguments that can be supplied to the program, with those beginning 
with '--' being optional arguments and the others being required arguments (with the exception of 'fn' argument not 
being necessary if 'ft' is 'DC')"""
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory to use so as to process the files contained within "
                                "them accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles' or 'NSAA'.")
parser.add_argument("ft", help="Specify type of file we wish to read from, being one of 'JA' (joint angle), "
                               "'AD' (all data), or 'DC' (data cube).")
parser.add_argument("fn", nargs="?", default="DC",
                    help="Specify the short file name to load; e.g. for file 'All-HC2-6MinWalk.mat' or "
                         "jointangleHC2-6MinWalk.mat, enter 'HC2'. Specify 'all' for all the files available "
                         "in the default directory of the specified file type. Optional for 'DC' file type.")
parser.add_argument("--dis_3d_pos", type=bool, nargs="?", const=True,
                    help="Plots the dynamic positions of an AllDataFile object over time. "
                         "Only works with 'ft' set as an 'AD' file name.")
parser.add_argument("--dis_diff_plot", type=bool, nargs="?", const=True,
                    help="Plots the diff plots of all features and axes of a JointAngleFile object over time."
                         "Only works with 'ft' set as a 'JA' file name.")
parser.add_argument("--dis_3d_angs", type=bool, nargs="?", const=True,
                    help="Plots the 3D positions of all features' joint angles over time. "
                         "Only works with 'ft' set as a 'JA' file name.")
parser.add_argument("--split_files", type=int,
                    help="Splits each of the files into '--split_files' number of parts to do statistical analysis on, "
                         "with each retaining the original file's file label ('HC' or 'D').")
parser.add_argument("--split_size", type=float, nargs="?", const=1,
                    help="Splits each of the files into multiple parts, with each part being of size in rows "
                         "'--split_size' x 'sampling_rate', so '--split_size' is the desired time frame in seconds for "
                         "the length of the split files (hence there are 'len of file in seconds' / '--split_size' "
                         "number of splits (i.e. '--split_size'=1 sets 60 frames (due to sampling rate) to each label)."
                         "Has no effect if '--split_files' called in same command.")
parser.add_argument("--check_for_abnormalities", type=float, nargs="?", const=True,
                    help="Checks for abnormalities within the currently-working-on file within it's parts. In other words,"
                         "if '--split_files=5', checks whether any of the 5 file parts is significantly different from "
                         "any of the other parts.")
parser.add_argument("--extract_csv", type=bool, nargs="?", const=True,
                    help="Directly writes a JA file from .mat to .csv format (i.e. without stat extraction).")
parser.add_argument("--del_files", type=bool, nargs="?", const=True,
                    help="Deletes the created file(s) as soon as they're created.")
parser.add_argument("--combine_all", type=bool, nargs="?", const=True,
                    help="Combines all the written files into a single file with sub-title containing '_ALL'")
parser.add_argument("--single_act", type=bool, nargs="?", const=True,
                    help="Specify if the files to operate on are 'single act' files.")
args = parser.parse_args()


#Gets default values for 'split_files' and 'split_size' if optional arguments are not provided
split_files = args.split_files if args.split_files else 1
split_size = args.split_size if split_files == 1 else None

#Section below sets 'local_dir' to the subdirectory that we shall be pulling .mat files from, along with setting
#up 'file_names' to be either the names of the files in subdirectory or a string if we are working with the data cube.
if args.dir + "\\" in sub_dirs:
    local_dir += args.dir + "\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within the source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'direct_csv', 'allmatfiles', or 'left-out'.")
    sys.exit()
file_names = []
if args.dir == "6minwalk-matfiles":
    if args.ft.upper() == "AD":
        local_dir += "all_data_mat_files\\"
        file_names = os.listdir(local_dir)
    elif args.ft.upper() == "JA":
        local_dir += "joint_angles_only_matfiles\\"
        file_names = os.listdir(local_dir)
    elif args.ft.upper() == "DC":
        local_dir += "joint_angles_only_matfiles\\"
        file_names = "DC"
elif args.dir == "6MW-matFiles":
    if args.ft.upper() == "AD":
        file_names = [f for f in os.listdir(local_dir) if f.endswith(".mat")]
    else:
        print("Second arg must be 'AD', as '6MW-matFiles' doesn't have joint angle or data cube files in them.")
        sys.exit()
elif args.dir == "left-out":
    file_names = [f for f in os.listdir(local_dir)]
else:
    if args.ft.upper() == "AD":
        # Only 'matfiles' subdirectory of 'NSAA' applicable for analysis with this script
        local_dir += "matfiles\\"
        if args.single_act:
            if args.dir == "NSAA":
                local_dir += "act_files\\"
            else:
                print("Must be using 'NSAA\\AD' files when using '--single_act' optional argument...")
                sys.exit()
        file_names = [f for f in os.listdir(local_dir) if f.endswith(".mat")]
    else:
        print("Second arg must be 'AD', as 'NSAA\matfiles' doesn't have joint angle or data cube files in them.")
        sys.exit()

#Only do the statistical analysis on files if none of the optional 'display' arguments have been given, as these do not
#require any statistical value extraction to display their results
if not args.dis_3d_pos and not args.dis_diff_plot and not args.dis_3d_angs:
    names = []
    if args.ft in file_types:
        if file_names != "DC":
            #Handles the 'AD'/'JA' case when file name is NOT 'all' (i.e. just a single file)
            if any(args.fn in fn for fn in file_names):
                file_name = local_dir + [fn for fn in file_names if args.fn in fn][0]
                names = class_selector(args.ft, file_name, fns=None, sub_dir=args.dir, is_all=False,
                                       split_files=split_files, is_extract_csv=args.extract_csv, split_size=split_size)
            elif args.fn == "all":
                file_names = [local_dir + fn for fn in file_names]
                names = class_selector(args.ft, None, fns=file_names, sub_dir = args.dir, is_all=True,
                                       split_files=split_files, is_extract_csv=args.extract_csv, split_size=split_size)
            else:
                print("Third arg ('fn') must be one of the file names for the '" + args.ft + "' file type, or 'all'")
                sys.exit()
        #Handles the 'data cube' case
        else:
            if not args.extract_csv:
                names = class_selector(args.ft, None, fns=None, sub_dir = args.dir, is_all=True,
                                       split_files=split_files, is_extract_csv=False, split_size=split_size)
            else:
                names = class_selector(args.ft, None, fns=None, sub_dir = args.dir, is_all=True,
                                       split_files=split_files, is_extract_csv=True, split_size=split_size)
    else:
        print("Second arg ('ft') must be one of the accepted file types ('JA', 'AD', or 'DC').")
        sys.exit()
    #Calls the 'check_for_abnormalities' function on the created .csv's if argument is specified
    if args.check_for_abnormalities:
        check_for_abnormality(names, error_margin=args.check_for_abnormalities)
    #Combines all the written output files into one file and deletes the individual files the 'ALL' was sourced from
    if args.combine_all:
        all_name = output_dir + args.dir + "\\" + args.ft + "\\" + args.ft + "_ALL_stats_features.csv"
        print("Combining all files into one and writing to " + all_name + "...")
        for i, name in enumerate(names):
            print("Adding", name, "to 'ALL' .csv...")
            df = pd.read_csv(name, index_col=0)
            if i == 0:
                with open(all_name, 'w', newline='') as file:
                    df.to_csv(file, header=True)
            else:
                with open(all_name, 'a', newline='') as file:
                    df.to_csv(file, header=False)
        #Get rid of the non-'ALL' files
        del_files(names)
    #If '--del_files' optional argument given, delete the files that have just been created
    if args.del_files:
        del_files(names)



#Final 3 optional arguments are called only if specified and, in the process, the script does not perform any
#of the above statistical analysis on the files given in argument; once the file object is created, its required
#display method is called
elif args.dis_3d_pos:
    if args.ft != "AD":
        print("Second arg ('ft') must be 'AD' for calling 'display_3d_positions' method.")
        sys.exit()
    else:
        if any(args.fn in fn for fn in file_names):
            file_name = local_dir + [fn for fn in file_names if args.fn in fn][0]
            AllDataFile(file_name, args.dir).display_3d_positions()
        else:
            print("Third arg ('fn') must be the short name of an all data file (e.g. 'D2', 'HC5').")
            sys.exit()
elif args.dis_diff_plot:
    if args.ft != "JA":
        print("Second arg ('ft') must be 'JA' for calling 'display_diffs_plot' method.")
        sys.exit()
    else:
        if any(args.fn in fn for fn in file_names):
            file_name = local_dir + [fn for fn in file_names if args.fn in fn][0]
            JointAngleFile(file_name, args.fn, args.dir).display_diffs_plot()
        else:
            print("Third arg ('fn') must be the short name of a joint angle file (e.g. 'D2').")
            sys.exit()
elif args.dis_3d_angs:
    if args.ft != "JA":
        print("Second arg ('ft') must be 'JA' for calling 'display_3d_angles' method.")
        sys.exit()
    else:
        if any(args.fn in fn for fn in file_names):
            file_name = local_dir + [fn for fn in file_names if args.fn in fn][0]
            JointAngleFile(file_name, args.fn, args.dir).display_3d_angles()
        else:
            print("Third arg ('fn') must be the short name of a joint angle file (e.g. 'D2').")
            sys.exit()
