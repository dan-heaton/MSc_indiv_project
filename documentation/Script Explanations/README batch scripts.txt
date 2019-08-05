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





------      Model prediction set scripts      ------

In an effort to make the execution of the model predictions sets easier (which often require numerous new 
models to be created with 'rnn.py' and many separate file predictions to be made with 'model_predictor.py'), 
we have created batch scripts to automate this process. This also holds the additonal benefit where any user 
can inspect what arguments we have run each script with and also enables them to run them for themselves to 
see if comparable results can be obtained (obviously requiring the setup of all other files via 
'comp_stat_vals.py' and other necessary scripts beforehand).

The idea is that, for each model prediction set that we are running, all that is needed is therefore to 
just run the specified '.cmd' script. This will build the requisite models though sometimes it won't build 
any new models but will instead rely on models built by previous '.cmd' scripts; hence, it's recommended 
that each model prediction set batch file is to be run in numerical ascending order. Once a given model 
prediction set batch file has been run, with the necessary models built and file predictions made, the 
results will appear in 'model_predictions.csv' as the final rows in the table. It's also worth noting the 
time discrepancies between some of the '.cmd' files: some will only be calling 'model_predictor.py' multiple 
times, which is comparatively quite quick to execute. However, those that call 'rnn.py' many times will take 
a lot longer; for example, 'model_predictions_set_3.cmd' needs to build 60 separate RNN models, each of 
which may take 10-15 minutes to run (assuming the user is building using a GPU), which could take 10-15 hours 
in total to execute the script.

Finally, the scripts don't take any arguments, as the Python script parameters have been decided in advance. 
For example, prior to executing model predictions sets 3 and up, we decided to test the models on the 
left-out subjects D3, D9, D11, D17, and HC6 (see the experiments results discussion set for an overview as 
to why these subjects were chosen). Hence, any changes that would be made to these '.cmd' scripts must 
modify each instance of the Python script that is called by the batch script in order to correctly alter 
these chosen script parameters.


