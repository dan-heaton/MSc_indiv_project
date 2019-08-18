---------------------		dis_3d_pos.py Overview and explanation		---------------------


------ Motivation ------

One desire for the data that we have received as '.mat' files is to be able to plot the subject portrayed 
within the file as a real-time 3D plot. The aim of this is to hopefully allow us to do two things:

	-- Visualize the subject within the data as doing certain activities in order to provide a reference 
	(along with the console 'Plotting time...' output) as to what activities are taking place at 
	which time
	-- In plotting this, easily allow for anomalies within the data file to be detected; for example, 
	if the subject suddenly 'jumps' position or the limbs appear extremely contorted, it might indicate 
	corrupted data which might need to be 'cut out' of the file (or have the whole file discarded)

Though this functionality also exists within the 'comp_stat_vals.py' script, it was felt necessary to also 
provide the functionality as a separate script within the system; hence, a lot of the code that was 
required by the '--dis_3d_pos' optional argument within 'comp_stat_vals.py' is repeated for this script.





------ How it works ------

This script involves a series of basic steps that the data goes through in order to display a dynamic, 3D 
plot to the user. Hence, we shall explain it here as these steps which include the following:

1.) Loads in a '.mat' file corresponding to the 'dir' and 'fn' arguments provided to the script. This is 
read in as a DataFrame object and is returned from 'preprocessing()' and passed to 'display_3d_positions()'.

2.) Extracts the values from the 'position' column and reads this in as a 'positions' matrix (of shape 
'# of samples' x '69'), separates the columns of this new matrix into tuples of x, y, and z axes for each 
segment within positions for every sample, define connected segments via tuples of pairs of values, sets 
the boarders of the 3D plot (i.e. the x/y/z mins/maxes), plots the 3D figure from the first sample with 
connections between points defined by the tuples of pairs of values, and animates it by fetching a new 
sample to plot every '1/sampling_rate' sections so the figure is animated in real-time while outputing to 
console the current time-stamp of the figure in seconds.

3.) After ~5 seconds where the data is sourced, extracted, reconfigured to work in 3D, and animated, a new 
window will appear. This is the 3D plot that runs in real time. Note that one should also see as a console output 
the time stamp in seconds of where the plot currently is at. There is no current way to pause, slow down, or 
speed up the plotting, though one can change the viewing perspective by left clicking and dragging with the 
cursor or zoom in and out by right clicking and dragging with the cursor.

