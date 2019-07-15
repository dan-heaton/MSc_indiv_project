---------------------		ext_raw_measures.py Overview and explanation		---------------------


------ Motivation ------

While the extraction of computed statistical values is an important tool for the data pipeline as an input to 
the RNN script, it's also necessary to be able to use different types of raw measurement values; in other words, 
the values that are recorded by the sensors of the body suit and are within the corresponding .mat files. For a 
given subject's suit data, each measurement (e.g. 'position', 'jointAngle', 'sensorMagneticField', etc.) is 
inserted into the .mat file's table of values as a column, with the height of the column equal to the number of 
samples that was taken of the subject (corresponding to the length of time the suit was recording x 60 samples 
per second). Within this single column, there are vectors of either 51, 66, or 69 values (depending on whether 
the suit was recording raw sensor values, angle values, or segment values, respectively).

The idea of this script is fairly simple. For a given subject name in a directory (or all the subject names found 
in that directory) and for a given measurement (or all raw measurements available), the relevant .mat file is 
opened, and the relevant column is expanded for the given measurement name so that it becomes a matrix of single 
values rather than a column of vectors (with a matrix of shape '# of samples' x '# of vector vals (e.g. 51, 66, 
or 69)'). This matrix of data is then to be written to a separate .csv file within a directory that reflects the 
source directory 'dir' and the measurement name that the matrix contains. 

From here, we can then use this data to train an RNN on these raw measurement values with y-labels (i.e. target 
values) that are determined by the type of file this .csv of data corresponds to (i.e. a 'D' or 'HC' subject), 
or the overall or single-act NSAA scores that correspond to the subject name of this .csv (e.g. 'D4') that can 
be found with 'nsaa_6mw_info'. In doing this, we provide an alternative to the production of RNN-ready data 
by 'comp_stat_vals' and are able to compare how manually extracted features differ in RNN performance to raw 
data (and thus the RNN doing its own feature extraction). This is explored further within the discussion of results.



------ How it works ------

The script runs in a fairly simple way without the necessity of classes or functions and thus just goes through 
a sequence of steps, which are as follows:

1.) Takes in arguments for the directory from which to retrieve the file(s) for raw measurement(s) extraction 
and checks them for validity (e.g. makes sure 'dir' is one of the allowed types).

2.) Retrieves the full file name(s) of the files within 'dir' from which we shall extract the measurements from. 
If 'fn'=all, retrieves all full file names in 'dir' as a list.

3.) Parse the list of measurements that we wish to extract based on the 'measurements' arg that are comma-
separated. If 'measurements'=all, then return a list of all extractable measurements available as a list.

4.) Create a directory for each raw measurement within 'dir' to store these raw measurements extracted as .csv's.

5.) For each file in 'dir', load the .mat file, extract the table of values within its tree structure, removes 
any 'wrappers' around these values within the table, and for each measurement to extract, select the column from 
the .mat table that correspond to the measurement, expand it out as 'measure_data', and write it to a .csv file 
that reflects the file name and measurement we are currently concerned with.
