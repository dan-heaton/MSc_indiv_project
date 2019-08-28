
---------------------		assess_nsaa_nmb_file.py Overview and explanation		---------------------


------ Motivation ------

As part of the finished deliverables for the project, we wanted to create a 'wrapper' Python script that 
was able to assess a single file (either an NSAA or NMB file) wherever it was located within a user's 
local system on the models that we have selected as our 'final' chosen models (i.e. those that are 
contained within '<project directory>\ source\rnn_models_final'). The idea of this script is that it would 
act as the primary tool that someone would use who only wants to assess a single file on the models that we 
have built and chosen as the best possible models for the job.

While we could have implemented this functionality as a batch script, it was felt that it would be easier 
implemented as a Python script. This made things like condition calling of over scripts, argument parsing, 
and so on easier than if we were using a batch script. It also allowed for dynamic user interaction through 
inputs to the script, which meant that the script could be written in a more user-friendly way. In other 
words, for this script we do away with taking in arguments and instead ask for user input at points 
throughout the script execution. The hope is that it makes it easier to use for any user and can simply 
run it with only the project directory obtained and a '.mat' file somewhere on their system of which they 
wish to assess.

As the script is meant to not require the local directory, this posed a potential problem to calling the 
other scripts (such as 'comp_stat_vals.py') which require the files to be located within the local directory 
in order to operate them. To get around this, we make use of the 'file_mover.py' functionality where, if it 
doesn't see a local directory where it's expecting (based on the 'local_dir' variable value from 
'settings.py'), as would be the case where the user doesn't have the local directory, it instead creates a 
local directory with the same name, with the corresponding inner directories, and places the file from 
wherever the user specified into here. This then allows the subsequence scripts to operate upon this file 
as normal.


------ How it works ------

The primary operation of the scipts is to take in user input and, from the various inputs, create strings 
that are passed in turn to the 'os.system()' function to call each script in turn with the correct 
arguments. Again, as this is a fairly simple script in its execution with no function calls or object 
creations, we can summarize the script as a series of several steps:

1.) Gets from the user the user-specified path to the '.mat' file which the script shall be assessing 
(can be either an absolute path or a path relative to the '<project directory>\source' directory.
2.) Gets from the user whether the file is of an NSAA assessment or a natural movement behaviour file.
3.) Executes 'file_mover.py' to move the file to the required subdirectory of the local directory (based on 
the NSAA\NMB choice specified) or, if the directory doesn't exist, creates the required directory and 
subdirectories and then moves it to the required subdirectory.
4.) Executes 'file_renamer' to renames the now-moved file if required.
5.) Gets from the user the comma-separated measurements (raw or computed statistical values) to use to 
assess the file.
6.) Executes 'ext_raw_measures.py' if required to extract the raw measurements from the file.
7.) Executes 'comp_stat_vals.py' and 'ft_sel_red.py' if required to extract the computed statistical values 
and reduced their dimensionality from the file.
8.) Gets from the user whether or not they wish to use models built on 'alt_dirs' and/or built solely 
on non-'V2' files.
9.) Based on the inputs given by the user regarding 'alt_dirs' and 'V2' files, execute 'model_predictor.py' 
to assess the file's measurement data on the appropriate models, display the results to console, and 
write the results to 'model_predictions_newfiles.csv'.