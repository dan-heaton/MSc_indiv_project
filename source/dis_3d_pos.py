import argparse
import pandas as pd
from settings import sub_dirs, local_dir, sampling_rate
import sys
import os
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from tkinter import*


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory to use so as to retrieve the file contained within "
                                "them accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles' or 'NSAA'.")
parser.add_argument("fn", type=str, help="Specify the short file name to load; e.g. for file 'All-HC2-6MinWalk.mat' or "
                                         "jointangleHC2-6MinWalk.mat, enter 'HC2'.")
args = parser.parse_args()



def preprocessing():
    """
        :return: Given the 'dir' and 'fn' arguments, checks for argument validity, loads the file from '.mat' form,
        and returns the complete file as a DataFrame
    """

    #Checks the 'dir' argument for validity (note: 'allmatfiles' and 'direct_csv' can't be use as these currently
    #only contain joint angle files and not the position data needed here)
    if args.dir + "\\" in sub_dirs and args.dir != "allmatfiles" and args.dir != "direct_csv":
        dir = args.dir
    else:
        print("First arg ('dir') not a valid directory. Must be within 'settings.sub_dirs'...")
        sys.exit()

    #Gets the file names of the '.mat' files for the corresponding 'dir' argument

    #Gets the correct path to the necessary directory as specified by the 'dir' argument
    if dir == "6minwalk-matfiles":
        dir_path = local_dir + dir + "\\all_data_mat_files\\"
    elif dir == "NSAA":
        dir_path = local_dir + dir + "\\matfiles\\"
    else:
        dir_path = local_dir + dir + "\\"

    #Gets the full name of the file within the 'dir' directory based on the 'fn' argument; exits the script if there
    #is no matching file containing 'fn'
    try:
        full_file_name = [dir_path + fn for fn in os.listdir(dir_path) if fn.endswith(".mat") and args.fn in fn][0]
    except IndexError:
        print("Second arg ('fn') must be the short name of the file to load within 'dir'...")
        sys.exit()

    #Loads the data from the given file name and extracts the root 'tree' from the file
    ad_data = sio.loadmat(full_file_name)
    #Corresponds to location in matlabfile at: 'tree.subjects.frames.frame'
    tree = ad_data["tree"]

    #Try-except clause here to catch when 'D6' doesn't have a 'sensorCount' category within 'tree.subject.frames'
    print("Extracting data from AD file " + full_file_name + "....")
    try:
        frame_data = tree[0][0][6][0][0][10][0][0][3][0]
    except IndexError:
        frame_data = tree[0][0][6][0][0][10][0][0][2][0]

    #Gets the types in each column of the matrix (i.e. dtypes of submatrices)
    col_names = frame_data.dtype.names

    #Extract single outer-list wrapping for vectors and double outer-list values for single values
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

    #With the newly extracted data from their wrappers, returns the data as a DataFrame
    return pd.DataFrame(frame_data, columns=col_names)



def display_3d_positions(df):
    """
        :param takes in the DataFrame object, 'df', of the 'dir' and 'fn' of the corresponding subject
        :return no return, but plots position values for the given file object on a 3D dynamic plot,
        with the necessary components connected, and outputs a dynamic accumulated time in seconds to the console:
    """
    #Disregard first 3 samples (usually types 'identity', 'tpose' or 'tpose-isb')
    positions = df.loc[:, "position"].values[3:]

    #Extracts position values for each dimension so 'xyz_pos' now has shape:
    ## of dimensions (e.g. 3 for x,y,z position values) x # of samples (~22K) x # of features (23 for segments)
    xyz_pos = [[[s for s in sample[i::3]] for sample in positions] for i in range(3)]

    #Xyz_pos_t now has dimensions:
    ## of features (23 for segments) x # of samples (~22K) x # of dimensions (e.g. 3 for x,y,z position values)
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

    xy_ratio = (maxs[1] - mins[1]) / (maxs[0] - mins[0])
    xz_ratio = (maxs[2] - mins[2]) / (maxs[0] - mins[0])
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
        if (i * 5) % sampling_rate == 0:
            print("Plotting time: " + str(int((i * 5) / sampling_rate)) + "s")
        i = (5 * i) % xyz_pos_t.shape[1]
        for line, pt, xi in zip(lines, pts, xyz_pos_t):
            x, y, z = xi[:i].T
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
        for stick_line, (sp, ep) in zip(stick_lines, stick_defines):
            stick_line._verts3d = xyz_pos_t[[sp, ep], i, :].T.tolist()
        fig.canvas.draw()
        if (i + 5) >= int(len(xyz_pos[0])):
            plt.close()
            print("Animation ended...")
        return lines + pts + stick_lines

    #Plot the figure in 3D space, with an update interval (defined in miliseconds) to be in real time
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xyz_pos[0]),
                                   interval=1000 / sampling_rate, blit=False, repeat=False)
    plt.show()



#Preprocesses the data to extract the data from the '.mat' file in question and passes it to have its
#position values plotted
display_3d_positions(preprocessing())
