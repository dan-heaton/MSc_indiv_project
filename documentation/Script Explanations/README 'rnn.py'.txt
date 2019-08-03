---------------------		rnn.py Overview and explanation		---------------------


------ Motivation ------

As the central element of the system insofar as it encompasses the learning and prediction models that 
are relied upon to produce the results, the importance of this script should be self-evident as it contains 
the class that defines the RNN's architecture, how it trains, predicts, and the instantiation and running 
of said class. Hence, rather than going through the motivation of writing this script or going through 
the basics of RNNs and their operation, we instead shall highlight a few important points about the 
structure of this RNN:

	-- We chose to use LSTM units instead of traditional neurons mainly due to their ability to learn 
	better and don't suffer the vanishing or exploding gradient problems as severely.
	-- Other hyperparameters within the RNN itself (number of layers, size of LSTM units, learning 
	rate, etc.) are kept as a constant throughout the experiments. These were found based on prior 
	'best practices' through prior research projects undertaken by others as well as rudimentary tuning 
	to find 'good enough' parameters.
	-- The final layer can be either a single node for classification, a single node for regression, 
	or 17 total nodes for single-act classification; hence, the building of the RNN model depends on the 
	arguments passed in to the script.
	-- The performance of the models that are built here are generally viewed by two means: the console 
	output at the end of the running of the 'rnn.py' script (which provides the info we need to fill in 
	the 'RNN Results.xlsx') or the 'model_predictor.py' script (which provides info for 
	'model_predictions.csv'). See the relevant README for more information of how 'model_predictor.py works.




------ How it works ------

The structure of the script is fairly complicated and slightly convoluted, with numerous conditional 
statements needed to handle various data processing edge cases and many possible optional argument 
combinations that sometimes interact with each other in strange ways that must be handled; hence rather 
thatn explaining the structure of the script in detail, it's instead worth going through how exactly the 
script works upon being instantiated from the command line with arguments. This should give the user the 
a good grasp of what's going on upon script instantiation:


1.) Reads in all required arguments (e.g. source directory, file name(s), output type, etc.) and optional 
args (e.g. sequence length, sequence overlap, leave out file choice, etc.) and checks each for validity.

2.) Preprocesses the data from the source directory and file name(s) chosen; this includes reading in all 
source '.csv' files, fetching the relevant 'y' labels for the 'x' data from the files, splitting the data 
into sequences, discarding a proportion of the sequences if necessary, splits into train/test components, etc.

3.) Builds the rnn object (instantiated from the 'RNN' class) with the necessary feature length, sequence 
length, size of LSTM units, number of hidden layers, and so on.

4.) Train the RNN on the 'x_train' and 'y_train' components, tests on the 'x_test and 'y_test' components.

5.) Prints out the performance on the test set to the console.

6.) Write to a .csv unique to this model the results of the predictions, the arguments used to run the 
script, and the results that were printed to the console output.


It's also worth touching on a few of the optional arguments. The required arguments should be 
self-explanatory and in no further need of elaboration; some of the optional arguments and what they are 
used for are covered in more detail in the experiments and results discussion for the project (such as 
'--seq_len', '--seq_overlap', and '--discard_prop'), but others aren't and so should be briefly touched 
upon here:

-- '--write_settings': This gives the user the option to store the results of the RNN that are printed 
to the output to the 'RNN Results.xlsx' file, rather than the user having to manually copy-paste console 
results to the file in a new row. This is generally used when new experiments with different RNNs are being 
carried out to save time and minimize the chances of human error.

-- '--create_graph': This will create a graph of the true values against the predicted values; as these 
are done in the continuous numerical domain, this is only really useful for the overall NSAA score output 
type and is generally written to a new file within the 'Graphs' directory to be used in results' discussions.

-- '--epochs': A quick way to modify the number of epochs needed to train a model; this over varies based on 
the type of file being trained; for example, stat values from 'AD' files generally need only about 20 epochs 
to converge, while we generally use >100 epochs for raw measurements. The epoch value therefore isn't kept 
as a constant like the other hyperparameters but rather fluctuates as necessary to achieve model convergence. 

-- '--other_dir': This argument is set with the name of another source directory in order to also include 
files from another directory (or directories) in order to train and test the model; it simply loads in 
additional files into the preprocessing function. The motiviation behind this is further explored in the 
results discussion of 'model_predictions.csv'.

-- '--leave_out': This is the standard way leave out a specific subject short name (e.g. 'D4') when training 
the model. This is the workaround instead of removing a subject from the source directory so the model is 
not exposed to the subject in the training process. This is primarily used in conjunction with 
'model_predictor.py' to test on the left-out subject in question for those particular models. See 
'model_predictions.csv' and its section in the results discussions for more information of using this arg.

-- '--balance': This is the way that we can either upsample or downsample the data set loaded in by calling 
the relevant functions within 'data_balancer.py'. The motiviation for rebalancing the data set and how it 
works is covered extensively in the README for that script and thus is not worth repeating here.


