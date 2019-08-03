---------------------		Batch scripts: Overview and explanation		---------------------


Along with the Python scripts that make up the system pipeline, we also make extensive use of several batch 
scripts for automating some of the tasks and for setup. As these aren't particularly long or complicated, 
it isn't worth creating a separate README for each, but rather a single README covering all of them along 
with when we would use them.


-- 'setup.cmd': this script runs the necessary 'pip' package installation commands to setup all the external 
libraries needed for running the project. Specific versions of the packages are used to match the exact 
versions used as part of this project to avoid potential complications, although setting up the most recent 
versions of the packages would most likely work just as well. We also run the necessary system scripts on 
all setup source directories. This requires that the user has setup the source directories ('NSAA', 
'6minwalk-matfiles', etc.) in a base directory that matches the name of the 'local_dir' global variable 
stored in 'settings.py'. Assuming that, the rest of 'setup' will extract the statistical values from each 
file in every directory, along with reducing the features of these, standardizing the names of the files, 
extracting all raw measurements from every 'AD' file, and dividing up files to extract single activities 
from 'AD' files.

-- 'model_pipeline': given the necessary arguments given along with the batch script, the 'comp_stat_vals.py', 
'ft_sel_red.py', and 'rnn.py' scripts are run in turn to assess the performance of the RNN model on 
extracted statistical values. Note that, as we generally will only run 'comp_stat_vals.py' once and at the 
beginning of the project's inception, this batch script isn't particularly required anymore and is kept in 
more for historical reasons.

-- 'file_predictor': this is the standard way that we wish to take in a new subject's file from 'inception' 
(i.e. given to us directly as a '.mat' file) and make predictions about it. This is done by first extracting 
the statistical values from the file via 'comp_stat_vals.py', extracting raw measurements via 'ext_raw_measures.py', 
reducing the feature space of stat values via 'ft_sel_red.py', and finally using all of these measures that 
were previously deemed (in the discussion of experiment results) to be 'useful' as measures (i.e. AD, 
jointAngle, sensorMagneticField, and position) and assess the file using these measures. While this process 
can be done just as easily by running each script in turn, this way is easier as it only requires us to 
run one script manually and with only two arguments (the source directory and the file name). Note that this 
is done on models that have already been built by 'rnn.py' and will try to use models that have the file left 
out of the training process, but if it can't find any then uses standard models trained on all files.

-- 'file_predictor_altdirs': very much similar to 'file_predictor' but takes an additional 'altdirs' arg 
as script input (for more info about this, please consult the README for 'test_altdirs.py' or the relevant 
results discussion concerning assessing alternative directories). With this argument, the only difference is 
that the 'model_predictor.py' script is called with the optional argument '--alt_dirs' set to the batch 
script argument.

-- 'file_predictor_leaveout': again, very similar to 'file_predictor', with the only difference being that, 
given the file name, models are built to distinctly leave out the file in question from any part of the 
model training. This ensures that the new file we are working with within this script is completely new to 
the models that will be predicting from in 'model_predictor.py'.






