

---------------------		comp_stat_vals.py Overview and explanation		---------------------

The basic operation of the 'comp_stat_vals.py' script can be summarized as follows:

1.) Read in a certain .mat file (either a joint angle file, an all-data file, or a datacube object) into Python

2.) Apply statistical analysis on its various features (i.e. each column of the 'table' of the .mat file (e.g. 
joint angle file has table of 22k x 66) (note that this is optional if a joint angle is selected and called with 
the 'write_direct_csv' method, which just translates a joint angle .mat file to .csv format)

3.) Write this out to a .csv file with a name corresponding to the read in file; the aim with this is for it to 
be an easy-to-digest format for the next stage in the analytics pipeline (e.g. a recurrent neural network)




------      Statistical extraction      ------

Each of the statistical analysis analyses that are used are implemented by a distinct function that performs a bit 
of syntactical help (e.g. the function to calculate mean includes type changing and rounding of numbers). The 
statistical features that are computed include: mean, variance, mean absolute diff values, FFT, covariance 
components between axes, mean sum of values across axes, among others.

The bulk of these features have been extracted 'intra-columns'. This is meant by the following: consider a 'JA' 
(joint angle) file; it's columns correspond to only 1 'measurement', the 'feature' itself ('feature' in this 
context meaning one of the 17 sensor labels, 22 joint labels, or 23 segment lables, the choice of which depends 
on which measurement we are referring to), and the 'dimension' of this feature (as we are dealing with 3D position 
data, this is 3D with each representing the 'X', 'Y' or 'Z' dimension). Many of the statistical features are thus 
computed on the values within each individual column; for example, the mean for a specific column is computed by 
averaging all the values for a specific measurement's specific feature's specific dimension (e.g. measurement 
'joint angle's feature 'jRightWrist's dimension 'X-dimension'), of which there are ~22k corresponding to ~22k 
samples in a file.

However, there are also several statistical functions that are applied one-layer up; that is, rather than 
calculating over a single column representing a single dimension of a feature of a measurement, it calculates over 
3 adjacent problems for ALL dimensions of a feature of a measurement. These mainly include operations that 
calculate features over a 2-dimensional array of data. Finally, the statistical functions that operate on single 
columns are then reapplied to the calculate the same statistical function over all newly-calculated values. For 
example, if we are concerned with the variance values for the 'position' measurement over 23 feature names over 
the 'x'-dimension, then we take the variance of these calculated values to form a new value representing the 
'position' measurement, the '(over all features)' feature, and the 'X'-dimension. This process is repeated with 
statistical functions that operate over the 3 axis dimensions.

The statistical features that are calculated per column (i.e. over a single axis) include:

- Mean
- Variance
- Absolute mean sample difference
- Fast Fourier Transform (1-dimension) largest val

The statistical features that are calculated per set of 3 columns (i.e. over all 3 axes of a feature for a 
given measurement) include:

- Mean sum of the values of each dimension
- Mean sum of the absolute values of each dimension
- First eigen value of the covariance matrix of the 3 columns
- Second eigen value of the covariance matrix of the 3 columns
- X- to Y-axis covariance (i.e. row 1 col 2 value of the 3x3 covariance matrix)
- X- to Z-axis covariance (i.e. row 1 col 3 value of the 3x3 covariance matrix)
- Y- to Z-axis covariance (i.e. row 2 col 3 value of the 3x3 covariance matrix)
- Fast Fourier Transform (2-dimension) largest 3 values (as 3 separate calculations)
- Proportion of samples outside the mean zone in every dimension


Each of these calculations done for a specific measurement, specific feature, and a single dimension is written 
as a single value as part of a row with the column title:
	"(<measurement name>) : (<feature name>) : (<axis>-axis) : (statistical function)"
...while, when it is subsequently called to repeat the process over all feature names, the column has the title:
	"(<measurement name>) : (over all features) : (<axis>-axis) : (statistical function)"

For the calculations done for a specific measurement, specific feature, and over all 3 dimensions, they are again 
written as a single value as part of a row with the column title:
	"(<measurement name>) : (<feature name>) : ((x,y,z)-axis) : (statistical function)"
...while, when it is subsequently called to repeat the process over all feature names, the column has the title:
	"(<measurement name>) : (over all features) : ((x,y,z)-axis) : (statistical function)"


The result is then a single row for a whole file with 'n' columns in the row, with each column corresponding to 
an above label. As we may have many measurements over which to calculate (e.g. 'position', 'velocity', 'angular 
acceleration', etc.), many features (e.g. 23, 22, or 17 depending on the measurement), 3 dimensions (or 1 dependent 
on which statistical function we are using), and ~15 statistical functions to compute, a single row for an 'AD' 
(all data) file can be several thousand columns long. Note that for a joint angle file this is significantly less 
as we are only concerned with 1 measurement (the 'jointAngle' measurement as this is the only one in the file).
	
Again note that these values are calculated across each of the samples (e.g. 22k) for each of the single columns 
or collection of 3 columns (depending on the statistical function in question).




------      Running this script      ------

To run this script and extract the desired .csv outputs, one must do the following (assuming one has access to 
the downloaded script already):

1.) Setup the required packages needed for the script (which are all 'included' at the beginning of the script), 
including numpy, scipy, pandas, etc (recommended to do so using 'pip' or 'conda' but manually downloading 
and putting them in 'Lib\Site-packages' is an option too)

2.) Change the initial global variable values for 'source_dir' and 'output_dir' to the location 
of the '6minwalk-matfiles' directory of JA and AD .mat files and to the location of the produced 
.csv files should go, respectively

3.) Open the data cube .mat file ('data_cube_6mw.mat') in MATLAB and run the 
'writetable(excel_table, "data_cube_table.csv)' command. This is necessary to extract the table within the datacube 
(which is needed in this script to process the datacube correctly) as Python is unable to read a .mat table in the 
same way as it can a .mat structure.

4.) Open up the command prompt (or terminal within an IDE), navigate to the scripts destination, and run the script 
with arguments for the file type (e.g. 'AD', 'JA', or 'DC) and a file name (either a specific short name like "D2" 
or "all" which processes all of the specified file type in the current directory); note that additional arguments 
are available (add '--help' to view these).

Hence, once the sufficient packages are installed, the global variables set, and the data cube table is extracted, 
the statistical feature values of all the AD files in the subdirectory '6minwalk-matfiles\all_data_mat_files' 
can be extracted by running the command

	"python matfiles_analysis.py ad all"

This may take a few minutes to complete, but once done, the file will appear in the output directory with the 
title 'AD_all_stats_features.csv', with each row corresponding to a single file and each column corresponding 
to the statistical values extracted from the file. Note that each file has the same measurements and features 
that are taken from their source .mat file, hence all the outputs to the .csv can share the same column labels.

The resultant file will then appear in the output directory specified by 'output_dir'.




------      Optional argument: split files functionality      ------

An additional argument can be given to the prompt as '--split_files=<split_val>' where 'split_val' is an integer.
This integer specifies that, for each file we are concerned with (e.g. all of them for a file type if 'all' is 
specified or just 1 if something like 'D2' is specified as a filename), we should split the source file samples 
into sections. For example, with a source 'AD' file with 22k samples (i.e. rows), if 'split_val' is specified to 
be '10', we divide this row into 10 parts essentially stacked on top of each other, where each part now has 2.2k rows.

The rest of the statistical extraction process follows as normal, with the exception that each of the statistical 
functions now operate on the file parts rather than the complete file itself; that's to say, for something like 
the mean of a single col's values, instead of computing the mean over 22k samples, it would compute 10 means of each 
part of length 2.2k samples.

This results in 10 rows of values that are outputed to a single .csv rather than just 1 value. Alternatively, if we 
specify something like 360 for 'split_val', then for .mat files that correspond to a time length of 360 seconds, then
each row in the output .csv file will correspond to the statistical features of a single second of the source .csv.
The idea is that this will give us more options when it comes to training ML models on this statistical data.




------      Optional argument: check for abnormalities      ------

This is a brief function that checks output .csv files for what seems to be abnormalities within the file. 
For example, say we have run the script on 'ad all --split_files=10'. We then have 10*number of AD files (= ~150)
rows in the 'AD_all_stats_features.csvs' file. This then (if the '--check_for_abnormalities=<margin>' optional 
argument is given) has each of its rows checked to see if any of its features (i.e. columns) fall outside the 
mean error margin given by 'ratio'; for example, if the mean value of a certain feature is '5' across all rows 
in the 'AD_all_stats_features.csvs' (i.e. over all file parts) and the 'margin' is set at 0.2, then a value for a
certain file part for that feature outside the range of 5-(0.2*5) and 5+(0.2*5) (i.e. 4 and 6) will be flagged up.

If enough of these flagged features occur (i.e. above a ratio set by 'abnormality threshold', which is defaulted 
to 0.3, meaning when more than 30% of features are beyond the mean error margin), then the offending file part 
will be printed to the console. The ideal is that, with a .csv containing many file parts for a given file (e.g. 
360 parts of a single file if it's split up into individual seconds), then we can easily observe strange behaviour 
given by the source data in the .mat file, which may be strange behaviour on the user's part or a fault in the 
data collection.




------      Optional argument(s): display various additional info      ------

There also exists 3 (currently) display functions that can infer information about the files in question that 
corresponds to the display optional argument in question (i.e. arguments 'ad D2 --dis_3d_pos' will call the 
'display_3d_positions' function on the 'AD' 'D2' file only). The behaviour of these are best observed first-hand 
rather than described, but as a brief summary:

- 'display_3d_positions': for a given 'AD' file, displays a 3D dynamic representation of an 'AD' file's position 
values (hence it won't work on a 'JA' or 'DC' file). This runs it real time, and thus will show a 3D figure walking
in 3D space over time

- 'display_3d_angles': for a given 'JA' file, displays a dynamic 3D representation of the joint angles as they 
change over time (note: this is somewhat deprecated and a specific use-case hasn't been formulated yet)

- 'display_diffs_plot': for a given 'JA' file, displays 66 total subgraphs on one plot (with 3 columns and 22 rows), 
with each row corresponding to a feature and each row corresponding to a feature's dimension (e.g. X, Y, or Z). Each 
graph themselves shows a plot of the diffs of the values of that feature's dimension (i.e. before statistical 
analysis is applied) as it changes over time 




