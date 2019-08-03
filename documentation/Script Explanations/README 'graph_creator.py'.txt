---------------------		graph_creator.py Overview and explanation		---------------------


------ Motivation ------

There are several different ways that the outputs of experiments can be stored as 'results', along with appearing 
in several locations. For example, the results of different RNN setups and its tests on the left-out test sets 
will appear in the 'RNN Results.xlsx' file. Additionally, for each model run (i.e. each row in 'RNN Results.xlsx'), 
there is a whole file of true and predicted values over the test stored stored in a single '.csv' file with a name 
that corresponds to the predictions. Meanwhile, the results of whole file predictions (i.e. through the use of 
'model_predictor.py' and it's wrapper script 'test_altdirs.py') are written as a row per file prediction into 
the 'model_predictions.csv'.

However, none of the scripts that write to these files do any sort of plotting or graphing of the data. This is 
for two reasons:

1.) Many times that we are running the scripts, we don't want to see the immediate plotting results or, rather, 
we can't. For example, when we run the 'model_predictor.py' script once, it's only concerned with writing a single 
line to 'model_predictions.csv', in the same way that 'rnn.py' only writes one line to 'RNN Results.xlsx', so for 
these to plot any results over several lines, the scripts would need additional user arguments to tell the script 
which lines it wishes to use for plotting, which adds to the already-high complexity of the scripts. Additionally, 
we often run 'rnn.py' via a batch script with many slight differences (to easily create a bunch of models to test 
on) and 'model_predictor.py' via 'test_altdirs.py', so stopping to produce a graph for every line that is written 
to an output file would be very inconvenient and would slow down the process.

2.) In separating the functionality, we keep a large degree of modularity amongst the scripts. In other words, the 
scripts that write the output to the output files ('model_predictions.csv', 'RNN Results.xlsx, etc.) have nothing 
to do with the actual plotting of results in graphs. This helps in debugging (i.e. a problem in displaying the 
data will be isolated to 'graph_creator.py') and also allows us to chose when we wish to do the plotting (i.e. 
after the data that we determine we need has been collected, not after a predetermined point in the running of 
each 'rnn.py' or 'model_predictor.py' run). Furthermore, this sort of setup opens up the possibility for an easier 
collaborative effort: if others were to contribute to the output files (e.g. by adding experiment results done on 
other types of data that is still written to the output files in the same format), then it's possible to use 
'graph_creator.py' as a standalone script without the need to have previously run any of the other scripts.





------ How it works ------

The direction that 'graph_creator.py' takes in terms of running entirely depends on the initial argument. Based on 
this, the script calls one of four functions that process the other given arguments in a certain way. Note that, 
as each function operates on the arguments given differently, some of them are given generic names such as 
'arg_one' and 'arg_two'. Also note that, as each function requires different numbers of arguments, every argument 
other than the first one ('choice') is optional; hence, when 'choice' is set to 'model_preds_single_acts', it 
won't throw an error when we only give it values for 'arg_one' and not the other three positional arguments.

Rather than going over things sequentially, we instead go over below each of the functions that are called by 
their associative 'choice' argument value:

-- plot_trues_preds() - this is a very simple function insofar as it just takes in the name of the '.csv' output 
that is produced by every run of 'rnn.py' that contains the test true and predicted values and are contained 
within the 'RNN_outputs' directory. Hence, the only argument needed is 'arg_one' and this is to be the full name 
(not including directories and the file extension) of the file we wish to use. This is then read in, the predicted 
and true values are read in, and these are plotted against each other in 2 dimensions, with a 'y=x' line going 
through them to signify their 'ideal' positions.

-- plot_model_preds_altdirs() - reads in the 'model_predictions.csv' as a DataFrame object; we then wish to 
determine which rows in the DataFrame object that we wish to use. This is then based on rows that have their 
'Source dir' column set to the value of 'arg_one' and the 'Model trained dir(s)' column set to the value of 
'arg_two'. For example, if we wish to plot the rows in 'model_predictions.csv' where a complete file from a 
specific source directory (e.g. 'allmatfiles') is then assessed on models trained on 'NSAA' and 
'6minwalk-matfiles' files, we set arg_one='allmatfiles' and arg_two='NSAA,6minwalk-matfiles'. This then selects the 
lines from the DataFrame object that we are concerned with. From here, with these lines we extract the true and 
predicted overall NSAA values from both the model trained on NSAA directory files and the model trained on 
6minwalk-matfiles directory files. These values are then plotted with the true values along the x-axis and the 
predicted values along the y-axis and is done for both models. We also extract the 'percentage of correctly 
predicted D/HC label for sequences' for each file and model this percentage distribution as both cumulative and 
non-cumulative distributions for both source directories. We then repeat the same process but for the columns 
representing the percentage of individual acts correctly determined, and finally plots some useful statistical 
values computed over the lines.

-- plot_model_preds_single_acts() - this is the second function to read in the 'model_predictions.csv' file, but 
as we treat 'single-act' rows in the file differently than those that use alternative directories for assessment, 
it's easier to keep the functionalities separated. Hence, we first load in the file as a DataFrame object and 
select only the rows that have '<args.arg_one> (act' in the name of the short file: this signifies that a 
single-act file has been assessed on a model, rather than a full source file. From here, for each line we extract 
from the row's cells the percentage of acts correctly predicted, the percentage of correctly predicted D/HC label 
for sequences, and difference between the true and predicted overall NSAA score. From these, we have take one 
of the value for each of the single-act files and plot these values against the act-number. This is then repeated 
2 more times for the other 2 extracted values over each of the 17 single-act files. This then leads to 3 subplots 
where the x-axis is the act number (between 1 and 17) and the y-axis is one of percentage of acts, correctly 
predicted, percentage of correctly predicted D/HC label for sequences, or diff between true/predicted overall NSAA.

-- plot_rnn_results() - this is the function that analyses the 'RNN Results' files and is responsible for the 
majority of graphs that show the performance of different RNN setups (e.g. sequence lengths, overlap proportions, 
number of features, types of raw measurements, etc.). The 'arg_one' and 'arg_two' args take the start and end 
experiment numbers of the file (once it has been loaded in as a DataFrame object) by looking at the 'Experiment 
Number' column to decide on which rows of the DataFrame object that we are concerned with. From here, for each 
row (which is associated with a model that has been created and tested upon), we extract the names of each 
measurement the model in question has used, the sequence length, and the results that it has produced. Then, based 
on the fourth provided arg ('xaxis_choice'), we decide on what to plot along the x-axis: if it's 'seq_length', then 
for each measurement (e.g. 'AD', 'jointAngle, etc.), we create a line and plot how well it performed at various 
sequence lengths with respect to different metrics (e.g. R^2, RMSE, etc.) based on the third provided arg 'out_type'.
If instead it's 'ft' (file type), 'seq_over' (sequence overlap), or 'features' (number of features used), then a 
single line to plot is used instead over all the lines from DataFrame selected to plot the aspect of the data 
specified by the 'xaxis_choice' against the metric specified by 'out_type'.



