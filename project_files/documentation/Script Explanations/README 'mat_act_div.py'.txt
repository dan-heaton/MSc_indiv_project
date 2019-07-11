---------------------		mat_act_div.py Overview and explanation		---------------------


------ Motivation ------

Along with using the full data files of the suit usage in various models and with varying target outputs (e.g. 
D/HC classification, overall NSAA score, etc.), we also wish to extract the single activities of .mat files 
from the NSAA directory. As standard, each .mat file in the NSAA directory contains the suit data of one full 
assessment for a single subject. This means that each file usually contains the subject performing all 17 
activities within the same file which are separated in time, sometimes by only a second or two in the case of 
the 'climb/descend box' activities and sometimes by up to a minute in the case of the 'get off the floor' 
activites. Hence, it would be advantageous for us to extract the data of the individual activites from within 
each file in order to use them for training.

As the data is contained within a very large table and each row is a single time instance of data (collected at 
60Hz from the suit, therefore each row is 1/60th of a second's worth of suit data), to create new single-activity 
files, all we need to do is the following:
	1.) Determine the start and end rows within the overall file of the activity in question (e.g. if we 
	wished to extract the second activity data that we know starts at 13s and ends at 15s in the subject's 
	assessment, we would need to extract rows 780 to 900 of the overall file)
	2.) Slice the relevant rows from the table and create a new '.mat'-friendly tree structure within 
	the script
	3.) Write this data to a 'act_files' subdirectory of the source directory as a new .mat file with a file 
	name reflecting which activity it represents

From here, we can process these single-activity .mat files in the same way as the standard .mat files, including 
extracting of raw measurements, computing of statistical values, and training of an RNN.


------ How it works ------

The key requirement for this script to work is by downloading the relevant Google document data sheet. This 
contains the manually assessed activity times of each subject, which was done by several members of the research 
group that analysed each of the videos that corresponds to each subject's .mat files and observed roughly 
at what times these activites started and ended for each subject. Note that these aren't going to be perfect, 
which is one flaw of using this sheet, as we can't give the exact start and end times of each activity and so 
tend to overestimate the amount of time the activity takes (i.e. note down a start time that's most likely before 
the real time and an end time that's after the true end time). Also, this process is not immune to human error, 
and therefore it's not impossible to misinterpret what what constitutes a 'complete' activity, which will impact 
how much use these 'single_act' files are for us.

This Google sheet that contains all these observations needs to be downloaded prior to running this script and 
can be found at "https://docs.google.com/spreadsheets/d/1OvkGU6kwmMxD6zdZqXcNKUvur1uFbAx5IND7_dXibjE" as a 
.csv to the relevant directory to the user (usually 'local_dir + "NSAA\\"'. From here, once this is read in by 
the script, there are two functions that are executed:

- 'extract_act_times'(): As the name suggests, this function analyses the Google sheet and from here, creates two 
lists: the first, 'act_times', is a list of start and end times (in suit frames, i.e. seconds x 60) in a nested 
structure (e.g. if there are 10 subjects, each performing the 17 activities, and each have a start and end time, 
then 'act_times' has a shape (10, 17, 2)); the second, 'ids', contain a list of subject names (e.g. 'D4'), each 
entry of which corresponds to an entry in 'act_times'.

- 'divide_mat_file': This function then takes the above two lists and, depending on what 'version' as an arg the 
user has specified ('V1' as standard; see the table itself for what the different versions correspond to) and 
what 'fn' arg the user has selected for the subject it whats to divide the activities of, the relevant row 
within 'act_times's is retrieved. The complete .mat file for the subject in question (determined by 'fn' arg) is 
then loaded, the table of data within the .mat file is extracted, and then, for each activity 'pair' (i.e. two 
numbers that are the start and end row numbers in the table for each activity), the table is sliced for that pair. 
These rows then 'replace' the rows of the 'whole' .mat file and the .mat file is then rewritten to a different 
file with a name reflecting the activity it is currently being concerned with in the 'for' loop. This then repeats 
for each of the 17 activites for the given 'fn' subject.

