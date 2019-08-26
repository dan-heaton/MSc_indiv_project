This is a brief overview of what is contained within each of the directories in this repo. For further information about 
this or the directory of raw and processed data (within the 'local directory'), please consult the 'Project and Local 
Directories Overview and Explanations' section of the final report.


Broadly speaking, the directories can be broken down as the following:

'background_research': this folder contains some preparatory programs that were written (with help from sources cited in the
scripts themselves) to familiarize oneself with the building of RNNs with the chosen libraries in order to make using them 
for the actual project later that much easier; hence, these aren't used by the rest of the project at all and are included 
for historical documentation reasons only.

'documentation': contains a number of documentation files pertaining to information about subjects that's needed for several 
scripts and also files that stores results from experimentation sets and model predictions sets. Consult either the 
'Project and Local Directories Overview and Explanations' or 'Reference Documents Explanations' section for further 
information about each file within 'documentation'.

	We also have several other subdirectories in ‘documentation’:

	'Graphs': all graphs created by 'graph_creator.py' are placed in here. These source from 'RNN Results.xlsx' and 
	'model_predictions.csv' to create graphs that are easier to display the results of groups of experiments done. 
	We see many of these graphs within the discussion of the experiment sets.

	'Script Explanations': collection of 'READMEs' for each of the scripts within 'project_files\source'. The idea is 
	that, if one wishes to find out what each script does, why it was written, etc., then reading its relevant 'README' 
	should provide sufficient detail. Much of these READMEs form the basis of our final report script overview later on.

'paper_reviews': contains paper reviews done of research papers that are believed to be relevant to the project. 
These predominantly focus on the use of RNNs when applied to real-world human movement data, and each paper consists of a 
slightly-shortened bullet pointed version of the paper and then a section of the most significant points from these bullet 
points. Hence, these papers are useful in justifying decisions taken with respect to model choices, experimentation 
directions, etc., and will also be heavily used in construction of the final project report.

'presentations': contains a collection of presentations that have been created to display to group members about the 
project's progress thus far (which are kept in order to be used in final report writings at a later date).

'report_stuff': contains several initial reports and other documentations of project progress thus far, and 
also 'MSc Project Plan.ods', which is where the already-completed and upcoming task lists are stored; this is particularly 
useful if one wishes to see what is currently being worked on or has recently been completed. The vast majority of the 
contents of this directory, however, is contained within this report.

'source': contains all of the scripts that are needed by the project pipeline to run. This includes the core Python scripts 
(such as 'comp_stat_vals.py' and 'rnn.py'), along with some 'supporting' scripts, such as 'settings.py' (to contain global 
variables that are used across several scripts). For information about how to run each of these scripts, run the script of 
interest through the command line/terminal with the ‘--help' optional argument set (e.g. ‘python comp_stat_vals.py --help'). This will display each of the arguments that are available to be set, the significance of each, how they interact with other arguments (if relevant), etc.

	'Batch files': Within this, we also contain the batch scripts that are used to automate some of the running of the 
	scripts. For further info about the significance of any or all of the scripts, consult the 'README(s)' for the 
	relevant scripts in 'Script Explanations', the script ecosystem overview in the final report, or the 
	diagram of the scripts and their connections to each other found in 'report_stuff'. Along with some of the simpler 
	automation of the tasks, we also run each of the model predictions sets from their respective batch files, as many 
	of them require building many models and testing many different combinations of files, which require many runs of 
	‘rnn.py’ and ‘model_predictor.py’; hence, the automation of this makes the process of replication hopefully much 
	easier for the user.