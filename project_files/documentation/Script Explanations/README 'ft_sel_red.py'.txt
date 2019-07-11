---------------------		ft_sel_red.py Overview and explanation		---------------------


------ Motivation ------

One of the consequences of using the 'comp_stat_vals' script is that the number of features for a single subject's 
all data ('AD') file baloons several fold: for a single subject with ~620 columns (with each being one feature 
of one measurement) and ~22K rows (360s at 60Hz suit sampling rate), this then becomes ~360 rows (given 
'--split_size'=1, i.e. 1 row for every 60 source rows) of approximately 4000 columns. Hence our data shape is 
now (360,4000) for a single file. This is completely impractical to use as training data for a given model for 
several reasons:
	1.) The curse of dimensionality means that the models struggle to train at all when dimensionality is 
	this large for the amount of data samples ('360') that we have available.
	2.) Many of these computed statistical features may hold not that much useful information in them, or at 
	least less useful information compared to other useful statistical features.
	3.) Even if we were to use all these features, it would take a much longer time to train models for most 
	likely very little gain (with it most likely being worse off than smaller dimensioned data), making it 
	even worse from a practical standpoint.

Hence, for the '_stat_features' files that are created by the 'comp_stat_vals' script, its more-or-less necessary 
to reduced the dimensionality to something a lot smaller prior to using this as training data. Note that this 
isn't done for raw measurement data. This is for three reasons:
	1.) The dimensionality of these data files is already at a level that is feasible for training (ranging 
	from 51 from sensor measurements to 69 for segment measurements)
	2.) There are far more rows of data within each of these files; this is due to the fact that, with using 
	'comp_stat_vals' with '--split_size'=1, we computed stat values over each block of 60 rows and hence 
	reduce the number of actual 'numbers of data' (i.e. numbers that appear in our data set) by 60 fold. This 
	60 fold increase in data when using raw measurements makes using this 51-69 dimensioned data a lot more 
	feasible in training models.
	3.) Even though we may be computing many redundant features in 'comp_stat_vals', we are much less likely 
	to have features that are as redundant as these in the raw measurements data. This is because every 
	feature corresponds to a single dimension for a sensor, angle, or segment, which is much more likely to 
	hold important information that many of the computed statistical values, and thus there is more of a 
	motivation to keep all of these.


------ How it works ------

Given a user-specified 'dir' for the directory that we wish to source the stat feature files from, the file type 
we're interested in (usually set to 'AD'), the 'fn' of the file(s) of which we wish to reduce the dimensions of 
(set to 'all' to do so over all files in 'dir'), and 'choice' (which is the feature selection/reduction 
technique to use), the following is undertaken by the script:

1.) For a given file name in 'dir', read in the file (e.g. 'AD_D4_stat_features.csv') as a DataFrame and divide 
it into it's 'x' and 'y' components.
2.) Normalize each dimension of the data if the relevant optional argument is set.
3.) Set the number of features to extract from the data if the relevant optional argument is set. As standard, we 
use '30', as this generally encompasses a vast amount of the variance inherent to each data file while also being 
a feasible data width for our RNN models.
4.) Based on the 'choice' argument given by the user, use a technique to reduce the dimensionality of the data. 
This can be done in an unsupervised feature dimensionality reduction manner (e.g. using principal component 
analysis or Gaussian random projection), unsupervised feature selection manner (e.g. variance thresholding or 
feature agglomeration), or in a supervised feature selection manner (e.g. by using a random forest for feature 
selection). 'PCA' has been used up until this point, though at the time of writing, this may be subject to further 
investigation and experimentation.
5.) With the newly-reduced data, call the 'add_nsaa_scores' function to add the overall and single-act NSAA 
scores to each of the rows of reduced-dimensionality data, which is necessary for getting the relevant 'y' labels 
by the 'rnn' script, which this script feeds into. The information for these scores comes from the 
'nsaa_6mw_info.xlsx' file, which contains the scores for every subject that has undertaken the NSAA assessment; 
hence, all that is required is to select the row in this .xlsx file that corresponds to the subject we are 
currently dealing with.
6.) The newly-reduced data, with the NSAA scores appended at the beginning of each row, is then written to the 
same directory as it was sourced, with the exception that an "FR_" ("feature reduced") prefix is appended to each 
file name to differentiate it from the file it came from.
7.) Repeat this process for every other file name in 'dir' that is required, which (if 'fn'=all results in all 
files in 'dir' having their dimensions reduced.

